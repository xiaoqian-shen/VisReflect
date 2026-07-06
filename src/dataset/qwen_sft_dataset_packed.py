import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import transformers
import ujson as json
from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
    VTK_TOKEN,
)
from src.params import DataArguments
from PIL import Image
from torch.utils.data import get_worker_info, IterableDataset
from transformers import TrainingArguments

from .data_utils import (
    get_image_info,
    get_vtk_crop_inputs,
    llava_to_openai,
    normalize_records,
    resolve_dataset_specs,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def make_packed_supervised_data_module(
    model_id, processor, data_args, training_args: TrainingArguments
):
    """Make dataset and collator for supervised fine-tuning."""

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()

    # data_path is either a direct records file or a meta-manifest (resolve_dataset_specs).
    loaded = json.load(open(data_args.data_path))
    specs = resolve_dataset_specs(loaded, data_args.image_folder)

    datasets = []
    total_data_len = 0
    for data, image_folder, ds_name in specs:
        iterable_sft_dataset = IterableSupervisedDataset(
            data_path=data,
            image_folder=image_folder,
            ds_name=ds_name,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=training_args.enable_data_packing,  # set for packed dataset
            random_seed=data_args.random_seed,
        )
        datasets.append(iterable_sft_dataset)
        total_data_len += len(iterable_sft_dataset)

    packed_train_dataset = PackedDataset(
        tokenizer=processor.tokenizer,
        datasets=datasets,
        # Get rank and world size from Hugging Face Trainer arguments
        data_rank=data_rank,
        data_world_size=data_world_size,
        # --- Configure your packing parameters ---
        max_packed_tokens=training_args.max_packed_tokens,
        max_buffer_size=100,
        # long_seq_cut=training_args.long_seq_cut,
        long_seq_threshold=training_args.long_seq_threshold,
        # Limiting the number of training data per device to avoid exploded tokens_per_device when pairing long with short
        max_instance_per_batch=training_args.max_instance_per_batch,
    )

    data_collator = PackedDataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=packed_train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    ), total_data_len


class IterableSupervisedDataset(torch.utils.data.IterableDataset):
    """
    An iterable version of your dataset that streams one processed sample at a time.
    This will be the input to PackedDataset.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        ds_name: str,
        model_id,
        data_rank=0,
        data_world_size=1,
        distributed_mode=True,  # set for packed dataset
        random_seed=None,
    ):
        super().__init__()
        if isinstance(data_path, str):
            self.raw_data = json.load(open(data_path, "r"))
        else:
            self.raw_data = data_path
        # Accept raw Visual-CoT (viscot_363k.json) records directly -- converted to the
        # internal {image, bboxes, conversations} format on load (no-op if already internal).
        self.raw_data = normalize_records(self.raw_data)

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.image_folder = image_folder
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.vtk_min_pixel = data_args.vtk_min_pixels
        self.vtk_max_pixel = data_args.vtk_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.area_threshold = data_args.area_threshold
        self.gen_tokens = data_args.gen_tokens
        self.ds_name = ds_name
        self.fps = data_args.fps

        self.data_world_size = data_world_size
        self.worker_id = None
        self.distributed_mode = distributed_mode
        self.worker_distributed = False
        self._state_dict = {}

        self.random_seed = None
        if random_seed:
            logger.info(f"{self.ds_name} is Shuffled!")
            self.random_seed = random_seed
            self.rng = np.random.default_rng(seed=self.random_seed)
            self.rng.shuffle(self.raw_data)

    def __len__(self):
        return len(self.raw_data)

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id :: self.num_workers]
            logger.info(
                f"worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}"
            )

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        assert self.worker_state_key is not None
        if (
            self.worker_state_key in self._state_dict
            and len(self._state_dict[self.worker_state_key]) > 0
        ):
            start_idx = self._state_dict[self.worker_state_key]["current_idx"]

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f"[{self.ds_name}] [Worker id {self.worker_id}] "
                f"begin to iter with {start_idx=}"
            )

        for i in range(start_idx, len(self.raw_data)):
            try:
                instance = self._build_instance(self.raw_data[i])
            except (ValueError, OSError) as e:
                # Skip records whose image/crop can't be processed -- corrupt or degenerate
                # data (0-dim image, truncated file, out-of-range crop) -- instead of
                # crashing the worker. Other exceptions (code bugs) still propagate.
                logger.warning(
                    f"[{self.ds_name}] skipping record {i} (bad image data): "
                    f"{type(e).__name__}: {e}"
                )
                continue
            yield instance

    def _build_instance(self, sources):
        """Process one record (a dict with ``image`` / ``bboxes`` / ``conversations``)
        into a packed-instance dict. Image(s) are read from local disk (each record path
        resolved against ``image_folder``)."""
        is_video = False

        processor = self.processor

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
            # Original full-res image is the bbox-crop source so the re-encoded crop
            # sees more detail than the downsized full image.
            pil = Image.open(image_file).convert("RGB")
            orig_images.append(pil)
            images.append(
                get_image_info(
                    pil,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                    image_patch_size=self.processor.image_processor.patch_size,
                )
            )

        # Re-encode each bbox crop; bboxes are assumed to index the first image.
        vtk_pixel_values, vtk_grid_thw, vtk_counts = get_vtk_crop_inputs(
            processor,
            orig_images[0],
            sources["bboxes"],
            self.vtk_min_pixel,
            self.vtk_max_pixel,
            self.area_threshold,
        )

        sources, vtk_counts = llava_to_openai(
            sources["conversations"],
            is_video=is_video,
            vtk_counts_list=vtk_counts,
        )
        sources = copy.deepcopy(sources)

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
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

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        # vtk crops are re-encoded inside the model; carry pixel values + grid.
        # vtk_lengths tracks crops-per-instance so packing can split correctly.
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vtk_pixel_values=vtk_pixel_values,
            vtk_grid_thw=vtk_grid_thw,
            vtk_lengths=torch.tensor([vtk_grid_thw.size(0)]),
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        data_dict["input_lengths"] = torch.tensor([input_ids.size(0)])

        return data_dict


"""
    Below is a adaptation of 
    https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset_packed.py
    We adopted a greedy data packing logic
"""

# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 4096,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
        # long_seq_cut: int = 25600,           # A single instance longer than this will be truncated
        long_seq_threshold: int = 6144,  # Instance longer than this will be individually processed
        max_instance_per_batch: int = 4,  # max num of instance per device
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens
        # self.long_seq_cut = long_seq_cut
        self.long_seq_threshold = long_seq_threshold
        self.max_instance_per_batch = max_instance_per_batch

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(
            VISION_START_TOKEN
        )
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self.vtk_token_id = self.tokenizer.convert_tokens_to_ids(VTK_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [len(d) for d in datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]

        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        print("#" * 42)
        print(f"Training with datasets and their weights:")
        for d, w in zip(datasets, self.dataset_weight):
            print(f"{d.ds_name}:\t{w}")
        print("#" * 42)

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            "sample_info": {d.ds_name: 0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), (
                f"deduplicated ds_name: {ds_name_list}"
            )

        for ds in self.datasets:
            self._state_dict[ds.ds_name] = {}

        if get_rank() == 0:
            logger.info(
                f"Loaded dataset to pack: {ds_name_list}, "
                f"{self.num_images_expected=}, {self.max_packed_tokens=}, "
                f"{self.replacement=}, {self.allow_overflow=}",
            )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f"{ds.ds_name:<25}: {ds_w * 100:.2f}%")
            temp = "\n".join(temp)
            logger.info(f"Sampling prob for each dataset:\n{temp}")

        if self.allow_empty_data:
            logger.warning(
                "allow_empty_data is enabled, note that empty data may be generated!"
            )

    def load_state_dict(self, state_dict, custom_infos=None):
        self.worker_custom_infos = custom_infos

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f"{ds.ds_name=} is resumed.")
            else:
                logger.warning(f"{ds.ds_name=} is not resumed.")

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                break
            except StopIteration:
                if self.replacement:
                    # logger.info(f'[Worker id {self.worker_id}] Dataset {self.datasets[current_dataset_idx].ds_name} is exhausted, restart it.')
                    try:
                        self.dataset_iter_list[current_dataset_idx] = iter(
                            self.datasets[current_dataset_idx]
                        )
                        current_sample = next(
                            self.dataset_iter_list[current_dataset_idx]
                        )
                        break
                    except:
                        # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)

                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)

                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except Exception as e:
                logger.error(
                    f"worker_id{self.worker_id} data_rank={self.data_rank} data_world_size={self.data_world_size} Unexpected error: {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                # Surface the real error instead of exit()-ing the worker, which the
                # caller's bare `except` used to swallow as "exhausted" -> 0 batches ->
                # train_loss=0. A genuine per-sample failure should be visible.
                raise

        current_ds_name = self.datasets[current_dataset_idx].ds_name

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop("meta_info", {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict["sample_info"][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        # NOTE: use `bisect` to search might be faster
        #  deleted the condition on # of images

        find = False
        find_idx = -1

        # if we see a new sample > LST, we need it to be in the buffer list,
        # instead of concatenating it to any existing buffer
        if new_sample["input_ids"].size(0) >= self.long_seq_threshold:
            return None

        for buffer_idx, buffer in enumerate(buffer_list):
            num_merged_tokens = new_sample["input_ids"].size(0) + buffer[
                "input_ids"
            ].size(0)
            num_instance_buffer = buffer["input_lengths"].size(0)
            if num_instance_buffer + 1 <= self.max_instance_per_batch:
                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if (
                    self.allow_overflow
                    and len(buffer_list) >= self.max_buffer_size // 2
                ):
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        else:
            return None

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample["data_index"] = torch.zeros_like(new_sample["input_ids"])
            return new_sample

        new_sample["data_index"] = (
            torch.ones_like(new_sample["input_ids"]) + buffer["data_index"][-1].item()
        )

        assert buffer.keys() == new_sample.keys()
        # All fields (incl. vtk_pixel_values/vtk_grid_thw/vtk_lengths) are tensors
        # concatenated along dim 0.
        for k in buffer:
            buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer

    @staticmethod
    def split_buffer(
        buffer,
        max_tokens,
        img_start_token_id,
        img_token_id,
        img_end_token_id,
        vtk_token_id,
        long_seq_threshold,
        max_instance_per_batch,
    ):
        if not long_seq_threshold:
            long_seq_threshold = max_tokens // 2

        def _image_is_splitted(input_ids, cut_idx):
            if cut_idx >= input_ids.size(0):
                return False
            else:
                is_image_start = input_ids[cut_idx].item() == img_start_token_id
                is_image_token = input_ids[cut_idx].item() == img_token_id
                is_image_end = input_ids[cut_idx].item() == img_end_token_id
                return is_image_start or is_image_token or is_image_end

        """
            Handles long single-/multi- instance buffer differently
        """
        # condition 1: single instance
        if buffer["data_index"][-1].item() == 0:
            # condition 1.1: single long instance
            if buffer["input_ids"].size(0) >= long_seq_threshold:
                """cut_id is the idx of the first token to be dropped"""
                cut_id = min(max_tokens, buffer["input_ids"].size(0))

                num_discarded_vtk_tokens = (
                    (buffer["input_ids"][cut_id:] == vtk_token_id).sum().item()
                    if vtk_token_id is not None
                    else 0
                )
                # Keep only if the cut neither splits an image nor drops any vtk
                # (crop) token; re-mapping dropped vtk tokens back to crops is not
                # worth it for the rare single-overlong case -> discard instead.
                if (
                    not _image_is_splitted(buffer["input_ids"], cut_id)
                    and num_discarded_vtk_tokens == 0
                ):
                    for k in buffer:
                        if k in [
                            "input_ids",
                            "labels",
                            "attention_mask",
                            "position_ids",
                            "data_index",
                        ]:
                            buffer[k] = buffer[k][:cut_id]
                        elif k in [
                            "pixel_values",
                            "image_flags",
                            "image_grid_thw",
                            "vtk_pixel_values",
                            "vtk_grid_thw",
                            "vtk_lengths",
                        ]:
                            buffer[k] = buffer[k]
                        elif k in ["input_lengths"]:
                            pass
                        else:
                            raise NotImplementedError(
                                f"find unsupported keys: {k} from {buffer.keys()}"
                            )
                    # re-assign lengths
                    buffer["input_lengths"][0] = buffer["input_ids"].size(0)
                    buffer_ready = [buffer]
                    buffer_unready = []
                else:  # if image/vtk is getting cut, discard the overlong instance
                    buffer_ready = []
                    buffer_unready = []

            # condition 1.2: single short instance
            else:
                buffer_ready = []
                buffer_unready = [buffer]

        # condition 2: multi instance
        else:
            # condition 2.1: < maxToken AND < max_instance_per_batch
            if (buffer["input_ids"].size(0) < max_tokens) and (
                buffer["input_lengths"].size(0) < max_instance_per_batch
            ):
                buffer_ready = []
                buffer_unready = [buffer]
            # condition 2.2: < maxToken AND == max_instance_per_batch
            elif (buffer["input_ids"].size(0) < max_tokens) and (
                buffer["input_lengths"].size(0) == max_instance_per_batch
            ):
                buffer_ready = [buffer]
                buffer_unready = []
            # condition 2.3: otherwise
            else:
                buffer_ready = []
                buffer_unready = []
                while buffer["input_ids"].size(0) >= max_tokens:
                    buffer_right = {}
                    cut_idx_right_size = buffer["input_lengths"][
                        -1
                    ].item()  # number of tokens to be cut from right side
                    image_cut_idx_right_size = int(
                        buffer["image_grid_thw"][-1].prod().item()
                    )  # number of image patches to be cut from right side
                    # crops of the last instance and their total vtk patch rows
                    nc = int(buffer["vtk_lengths"][-1].item())
                    vtk_pixel_cut = (
                        int(buffer["vtk_grid_thw"][-nc:].prod(dim=1).sum().item())
                        if nc > 0
                        else 0
                    )
                    for k in buffer:
                        if k in [
                            "input_ids",
                            "labels",
                            "attention_mask",
                            "position_ids",
                            "data_index",
                        ]:
                            buffer_right[k] = buffer[k][-cut_idx_right_size:]
                            buffer[k] = buffer[k][:-cut_idx_right_size]
                        elif k in ["pixel_values", "image_flags"]:
                            buffer_right[k] = buffer[k][-image_cut_idx_right_size:]
                            buffer[k] = buffer[k][:-image_cut_idx_right_size]
                        elif k in ["vtk_pixel_values"]:
                            if vtk_pixel_cut > 0:
                                buffer_right[k] = buffer[k][-vtk_pixel_cut:]
                                buffer[k] = buffer[k][:-vtk_pixel_cut]
                            else:
                                buffer_right[k] = buffer[k][0:0]
                        elif k in ["vtk_grid_thw"]:
                            if nc > 0:
                                buffer_right[k] = buffer[k][-nc:]
                                buffer[k] = buffer[k][:-nc]
                            else:
                                buffer_right[k] = buffer[k][0:0]
                        elif k in ["image_grid_thw", "input_lengths", "vtk_lengths"]:
                            buffer_right[k] = buffer[k][-1:]
                            buffer[k] = buffer[k][:-1]
                        else:
                            raise NotImplementedError(
                                f"find unsupported keys: {k} from {buffer.keys()}"
                            )
                    # if left buffer is longer
                    if buffer["input_ids"].size(0) >= buffer_right["input_ids"].size(0):
                        buffer_ready.append(buffer)
                        buffer = buffer_right
                    else:  # buffer_right is longer than the accumulated left
                        buffer_ready.append(buffer_right)

                # if buffer['input_ids'].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                buffer_unready.append(buffer)

        return buffer_ready, buffer_unready

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        # NOTE: in-place operation

        buffer_ready, buffer_unready = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
            vtk_token_id=self.vtk_token_id,
            long_seq_threshold=self.long_seq_threshold,
            max_instance_per_batch=self.max_instance_per_batch,
        )

        for each_buffer in buffer_ready:
            buffer_max_len_list.append(each_buffer)

        for each_buffer in buffer_unready:
            find_idx = len(buffer_list)
            num_tokens_new_sample = each_buffer["input_ids"].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]["input_ids"].size(0) < num_tokens_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        return buffer_list, buffer_max_len_list

    def print_log(self, iter_idx, buffer_list):
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(
                f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}"
            )

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f"Begin to iter, {len(buffer_list)=}")

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f"work_state_{self.worker_id}"
        self.datasets = [d for d in self.datasets_orig]

        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            # if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
            ds.worker_id = worker_id
            ds.worker_state_key = f"work_state_{self.worker_id}"
            ds.num_workers = num_workers
            if self._should_log() and worker_id == 0:
                logger.info(
                    f"set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}"
                )

        if (
            self.worker_custom_infos is not None
            and self.worker_state_key in self.worker_custom_infos
        ):
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if "buffer_list" in custom_infos and isinstance(
                custom_infos["buffer_list"], list
            ):
                buffer_list = custom_infos["buffer_list"]
                if self._should_log() and worker_id == 0:
                    logger.info(
                        f"[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}"
                    )
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(
            f"{self.__class__.__name__} Rank {self.data_rank} "
            f"Worker {worker_id} begin to load data"
        )

        while True:
            self.dataset_weight = [
                w / sum(self.dataset_weight) for w in self.dataset_weight
            ]
            current_dataset_idx = rng.choice(
                len(self.dataset_iter_list), p=self.dataset_weight
            )

            try:
                current_sample = self.next_data(current_dataset_idx)
            except StopIteration:
                # Only true exhaustion flushes and ends the stream; real per-sample
                # errors propagate (visible) instead of masquerading as "exhausted".
                logger.info(
                    f"All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})"
                )
                while len(buffer_list) > 0:
                    yield buffer_list.pop(0)
                logger.info(f"buffer_list is empty! ({len(buffer_list)=})")
                return

            # it is guaranteed in self.find_buffer() that if current_sample is >= max_tokens,
            # it will not get concatenated to any existing buffer
            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            """
                A greedy method to balance effifiency and memory safety:
                1. if buffer stacks up tp max_packed_tokens, it is poped
                2. if buffer has >= max_instance_per_batch, it is poped
                3. if a single sample is >= long_seq_thresh, it is poped as a single buffer

                This is intended to avoid a long seq paired with multiple short seqs since
                padding them will explode the memory
            """
            buffer_list, buffer_max_len_list = self.update_buffer_list(
                buffer_list, buffer_max_len_list, buffer
            )

            while len(buffer_max_len_list) > 0:
                yield buffer_max_len_list.pop(0)

            while len(buffer_list) > self.max_buffer_size:
                yield buffer_list.pop(0)

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1


class PackedDataCollatorForSupervisedDataset:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # features is supposed to be a list of packed items
        # We will set batch_per_device to 1

        if not isinstance(features, list):
            features = [features]

        #  Unpack all sequences
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        all_vtk_pixel_values = []
        all_vtk_grid_thw = []
        all_pixel_values = []
        all_image_grid_thw = []

        for feature in features:
            # each feature is a packed mini-batch
            all_input_ids.extend(
                torch.split(feature["input_ids"], feature["input_lengths"].tolist())
            )
            all_attention_masks.extend(
                torch.split(
                    feature["attention_mask"], feature["input_lengths"].tolist()
                )
            )
            all_labels.extend(
                torch.split(feature["labels"], feature["input_lengths"].tolist())
            )

            all_vtk_pixel_values.append(feature["vtk_pixel_values"])
            all_vtk_grid_thw.append(feature["vtk_grid_thw"])
            all_pixel_values.append(feature["pixel_values"])
            all_image_grid_thw.append(feature["image_grid_thw"])

        # pad all sequences
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        for i in range(len(all_input_ids)):
            seq_len = len(all_input_ids[i])
            padding_needed = max_len - seq_len

            # Pad on the right side.
            padded_input_ids.append(
                torch.nn.functional.pad(
                    all_input_ids[i], (0, padding_needed), value=self.pad_token_id
                )
            )

            padded_attention_masks.append(
                torch.nn.functional.pad(
                    all_attention_masks[i], (0, padding_needed), value=0
                )
            )

            padded_labels.append(
                torch.nn.functional.pad(
                    all_labels[i], (0, padding_needed), value=IGNORE_INDEX
                )
            )

        # vtk_tokens carries the re-encode crop pixels; vtk_grid their grids.
        # The model forward consumes them via get_image_features(vtk_tokens, vtk_grid).
        data_dict = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
            "vtk_tokens": torch.cat(all_vtk_pixel_values),
            "vtk_grid": torch.cat(all_vtk_grid_thw),
            "pixel_values": torch.cat(all_pixel_values),
            "image_grid_thw": torch.cat(all_image_grid_thw),
        }

        return data_dict
