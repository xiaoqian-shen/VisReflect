"""Non-streaming, local-JSON video SFT dataset for VisReflect (visual-reflection on video).

The training data is a plain local JSON list -- no Manifold, no streaming. Each record is::

    {"video": "<path rel to --video_folder or absolute>",
     "question": "<question text>",
     "answer": "<answer text>",
     "temporal_span": [lo_sec, hi_sec]}   # or a list of [lo, hi] pairs (multiple clue spans)

``num_clue_frames`` frames are uniformly sampled from the clue span, re-encoded through the
ViT as the ``<VR>`` alignment target (the video analogue of the image bbox crop); the full
video is decoded as the question context. One video = one sequence (no packing).
"""

from __future__ import annotations

import copy
import logging
import os

import decord
import torch
import transformers
import ujson as json
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from torch.utils.data import Dataset
from transformers import TrainingArguments

from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    LLAVA_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)
from src.params import DataArguments

from .data_utils import (
    build_video_reflection_answer,
    fetch_video,
    replace_image_tokens,
    sample_clue_frame_indices,
)

logger = logging.getLogger(__name__)


def build_video_instance(
    vr: "decord.VideoReader",
    clue_intervals: list,
    clean_question: str,
    clean_answer: str,
    processor: transformers.ProcessorMixin,
    data_args: DataArguments,
    max_seq_len: int,
) -> dict:
    """Turn one decoded video + its clue span + QA into a packed-instance dict.

    Uniformly samples ``num_clue_frames`` frames from the clue span as the ``<VR>`` alignment
    target (``vtk_tokens`` / ``vtk_grid``), decodes the full video as the question context
    (``pixel_values`` / ``image_grid_thw`` -- the collator renames these to the video keys),
    and tokenizes a single-turn conversation with the reflection block prepended to the answer.
    """
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    image_processor = processor.image_processor
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size

    # --- alignment target: uniformly sample clue frames, encode as images ---
    frame_idx = sample_clue_frame_indices(
        clue_intervals, video_fps, total_frames, data_args.num_clue_frames
    )
    clue_np = vr.get_batch(frame_idx).asnumpy()  # [N, H, W, C]
    rh, rw = smart_resize(
        clue_np.shape[1],  # height
        clue_np.shape[2],  # width
        factor=patch_size * 2,
        min_pixels=data_args.vtk_min_pixels,
        max_pixels=data_args.vtk_max_pixels,
    )
    clue_frames = [
        Image.fromarray(clue_np[i]).resize((rw, rh)) for i in range(len(clue_np))
    ]
    # Encode the frames straight through the image processor (like get_vtk_crop_inputs for
    # image crops) -- the full processor would expect matching <image> tokens in the text.
    vtk_out = image_processor(images=clue_frames, do_resize=False, return_tensors="pt")
    vtk_pixel_values = vtk_out["pixel_values"]
    vtk_grid = vtk_out["image_grid_thw"]
    # one <VR> per merged crop token, across all sampled frames
    num_gen_tokens = int(vtk_grid.prod(dim=1).sum().item()) // (merge_size**2)

    # --- full video input (the question context) ---
    videos, video_metadata = fetch_video(
        vr,
        data_args.video_min_pixels,
        data_args.video_max_pixels,
        patch_size,
        max_seq_len,
        fps=data_args.fps,
        max_frames=data_args.max_frames,
    )

    # --- build the conversation (reflection block before the answer) ---
    role_map = {"human": "user", "gpt": "assistant"}
    conversations = [
        {"from": "human", "value": f"{LLAVA_VIDEO_TOKEN}\n{clean_question}"},
        {
            "from": "gpt",
            "value": build_video_reflection_answer(clean_answer, num_gen_tokens),
        },
    ]
    sources = copy.deepcopy(
        [
            {
                "role": role_map[c["from"]],
                "content": replace_image_tokens(c["value"], is_video=True),
            }
            for c in conversations
        ]
    )

    all_input_ids = []
    all_labels = []
    all_pixel_values = []
    all_video_grid_thw = []
    all_second_grid = []

    if len(SYSTEM_MESSAGE) > 0:
        system_message = (
            f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        )
        system_ids = processor.tokenizer(
            system_message, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        all_input_ids.append(system_ids.squeeze(0))
        all_labels.append(torch.full_like(system_ids, IGNORE_INDEX).squeeze(0))

    for j in range(0, len(sources), 2):
        user_input = sources[j]
        gpt_response = sources[j + 1]

        user_text = (
            f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}"
            f"{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
        )
        gpt_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

        if DEFAULT_VIDEO_TOKEN in user_text:
            # Qwen processor expects `videos` as a list of clips (one 4D T,C,H,W tensor each),
            # matching the eval path's process_vision_info output.
            inputs = processor(
                text=[user_text],
                images=None,
                videos=[videos],
                video_metadata=[video_metadata],
                padding=False,
                do_resize=False,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_input_ids = inputs["input_ids"]
            all_pixel_values.append(inputs["pixel_values_videos"])
            all_video_grid_thw.append(inputs["video_grid_thw"])
            if "second_per_grid_ts" in inputs:
                spgt = torch.as_tensor(
                    inputs["second_per_grid_ts"], dtype=torch.float32
                ).flatten()
            else:
                spgt = torch.ones(inputs["video_grid_thw"].size(0))
            all_second_grid.append(spgt)
        else:
            prompt_input_ids = processor.tokenizer(
                user_text, add_special_tokens=False, padding=False, return_tensors="pt"
            )["input_ids"]

        response_input_ids = processor.tokenizer(
            gpt_text, add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
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

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        vtk_tokens=vtk_pixel_values,
        vtk_grid=vtk_grid,
        # The collator renames pixel_values/image_grid_thw -> the video keys.
        pixel_values=torch.cat(all_pixel_values, dim=0),
        image_grid_thw=torch.cat(all_video_grid_thw, dim=0),
        second_per_grid_ts=torch.cat(all_second_grid),
        input_lengths=torch.tensor([input_ids.size(0)]),
    )


def _clue_intervals(rec: dict) -> list:
    """Normalize a record's ``temporal_span`` (a single ``[lo, hi]`` pair or a list of pairs;
    ``clue_intervals`` is also accepted) into a list of ``[lo, hi]`` second-intervals."""
    span = rec.get("temporal_span", rec.get("clue_intervals"))
    if span is None:
        raise ValueError("video record missing 'temporal_span' (or 'clue_intervals')")
    if len(span) == 2 and all(isinstance(x, (int, float)) for x in span):
        return [[float(span[0]), float(span[1])]]
    return [[float(lo), float(hi)] for lo, hi in span]


class VideoSupervisedDataset(Dataset):
    """Map-style local-JSON video dataset. `__getitem__` opens the video (decord) and builds
    one packed-instance dict; a bad video is skipped by advancing to the next record."""

    def __init__(
        self,
        data_path,
        video_folder,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        max_seq_len: int,
    ):
        super().__init__()
        self.records = (
            json.load(open(data_path)) if isinstance(data_path, str) else data_path
        )
        self.video_folder = video_folder
        self.processor = processor
        self.data_args = data_args
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def _resolve(self, video_path: str) -> str:
        if (
            self.video_folder
            and not os.path.isabs(video_path)
            and not os.path.exists(video_path)
        ):
            return os.path.join(self.video_folder, video_path)
        return video_path

    def __getitem__(self, i):
        n = len(self.records)
        for k in range(n):
            idx = (i + k) % n
            rec = self.records[idx]
            try:
                vr = decord.VideoReader(self._resolve(rec["video"]), num_threads=1)
                return build_video_instance(
                    vr,
                    _clue_intervals(rec),
                    rec["question"],
                    rec["answer"],
                    self.processor,
                    self.data_args,
                    self.max_seq_len,
                )
            except Exception as e:  # noqa: BLE001 -- one bad video must not kill training
                logger.warning(
                    "skipping video record %d (%s): %s", idx, rec.get("video"), e
                )
        raise RuntimeError("no loadable video records in the dataset")


class PackedDataCollatorForSupervisedDatasetVideo:
    """Pad a batch of video instances. Each feature is one video (one sequence);
    vtk_tokens/vtk_grid are the clue-frame alignment crops, pixel_values/image_grid_thw are
    the full-video frames (renamed here to the keys the Qwen2.5-VL forward expects)."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        if not isinstance(features, list):
            features = [features]

        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_vtk_tokens = []
        all_vtk_grid = []
        all_pixel_values = []
        all_video_grid_thw = []
        all_second_per_grid_ts = []

        for feature in features:
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
            all_vtk_tokens.append(feature["vtk_tokens"])
            all_vtk_grid.append(feature["vtk_grid"])
            all_pixel_values.append(feature["pixel_values"])
            all_video_grid_thw.append(feature["image_grid_thw"])
            all_second_per_grid_ts.append(feature["second_per_grid_ts"])

        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        for i in range(len(all_input_ids)):
            pad = max_len - len(all_input_ids[i])
            padded_input_ids.append(
                torch.nn.functional.pad(
                    all_input_ids[i], (0, pad), value=self.pad_token_id
                )
            )
            padded_attention_masks.append(
                torch.nn.functional.pad(all_attention_masks[i], (0, pad), value=0)
            )
            padded_labels.append(
                torch.nn.functional.pad(all_labels[i], (0, pad), value=IGNORE_INDEX)
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
            "vtk_tokens": torch.cat(all_vtk_tokens),
            "vtk_grid": torch.cat(all_vtk_grid),
            "pixel_values_videos": torch.cat(all_pixel_values),
            "video_grid_thw": torch.cat(all_video_grid_thw),
            "second_per_grid_ts": torch.cat(all_second_per_grid_ts),
        }


def make_supervised_data_module_video(
    model_id, processor, data_args, training_args: TrainingArguments
):
    """Local-JSON video train dataset + collator. ``max_packed_tokens`` is the per-video max
    sequence length (fetch_video spreads the budget across frames)."""
    train_dataset = VideoSupervisedDataset(
        data_path=data_args.data_path,
        video_folder=data_args.video_folder,
        processor=processor,
        data_args=data_args,
        max_seq_len=training_args.max_packed_tokens or 16384,
    )
    data_collator = PackedDataCollatorForSupervisedDatasetVideo(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
