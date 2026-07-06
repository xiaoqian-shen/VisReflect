import copy
import os
from typing import Dict

import torch
import transformers
import ujson as json
from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    SYSTEM_MESSAGE,
)
from src.params import DataArguments
from PIL import Image
from torch.utils.data import Dataset

from .data_utils import (
    get_image_info,
    get_vtk_crop_inputs,
    llava_to_openai,
    normalize_records,
    pad_sequence,
    resolve_dataset_specs,
)


class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str | list,
        image_folder: str,
        ds_name: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        # Accept raw Visual-CoT (viscot_363k.json) records directly -- converted to the
        # internal {image, bboxes, conversations} format on load (no-op if already internal).
        self.list_data_dict = normalize_records(list_data_dict)
        self.data_args = data_args
        self.image_folder = image_folder
        self.ds_name = ds_name
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.vtk_min_pixel = data_args.vtk_min_pixels
        self.vtk_max_pixel = data_args.vtk_max_pixels
        self.area_threshold = data_args.area_threshold
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """This code currently assumes single image + multi/single bboxes"""

        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            orig_images = []

            for image_file in image_files:
                if not os.path.exists(image_file) and not image_file.startswith("http"):
                    image_file = os.path.join(image_folder, image_file)
                orig_images.append(Image.open(image_file).convert("RGB"))
                images.append(
                    get_image_info(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h,
                        image_patch_size=self.processor.image_processor.patch_size,
                    )
                )
        else:
            grid_key = None
            pixel_key = None
            images = None
            orig_images = None
            videos = None

        # Re-encode each bbox crop as the alignment target (bboxes index image 0).
        bboxes = sources["bboxes"]
        vtk_pixel_values, vtk_grid_thw, vtk_counts = get_vtk_crop_inputs(
            self.processor,
            orig_images[0],
            bboxes,
            self.vtk_min_pixel,
            self.vtk_max_pixel,
            self.area_threshold,
        )

        sources, _ = llava_to_openai(
            sources["conversations"],
            is_video=is_video,
            vtk_counts_list=vtk_counts,
        )
        sources = copy.deepcopy(sources)

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for j in range(0, len(sources), 2):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(
                    user_input,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors="pt",
                )["input_ids"]

            # filling the response with bboxes

            response_input_ids = processor.tokenizer(
                gpt_response,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"]

            input_ids = torch.cat(
                [prompt_input_ids, response_input_ids], dim=1
            ).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # No eos/bos tokens in input_ids -- Qwen2-VL does not use them.
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = torch.ones_like(input_ids)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vtk_pixel_values=vtk_pixel_values,
            vtk_grid_thw=vtk_grid_thw,
        )

        if pixel_key and grid_key:
            data_dict[pixel_key] = torch.cat(all_pixel_values, dim=0)
            data_dict[grid_key] = torch.cat(all_image_grid_thw, dim=0)

        return data_dict


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side="right", padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(
            batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX
        )

        vtk_pixel_values = torch.cat(
            [example["vtk_pixel_values"] for example in examples], dim=0
        )
        vtk_grid = torch.cat([example["vtk_grid_thw"] for example in examples], dim=0)

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "vtk_tokens": vtk_pixel_values,
            "vtk_grid": vtk_grid,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            # pixel_values_raw = batch_pixel_values   # Now its a list of pixel values
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            # data_dict["pixel_values_raw"] = pixel_values_raw
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict


def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    loaded = json.load(open(data_args.data_path))
    specs = resolve_dataset_specs(loaded, data_args.image_folder)
    data, image_folder, ds_name = specs[0]
    sft_dataset = SupervisedDataset(
        data_path=data,
        image_folder=image_folder,
        ds_name=ds_name,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator
    )
