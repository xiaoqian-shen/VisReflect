from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    loss_align_lambda: float = field(default=1e-1)

    # VisReflect video (Option A): compute the CE loss on a SECOND LM pass whose <VR> slots are
    # fed the model's own (detached) pre-<VR> hidden state -- the latent reflection eval uses --
    # instead of the real clue-frame embeddings (which stay ONLY as the loss_align target). Only
    # train_video.py propagates this to model.config; image training keeps the single-pass path.
    vtk_train_latent_ce: bool = field(default=True)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    # Per-parameter-group learning rates (create_optimizer); None -> use the base learning_rate.
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None

    checkpoint_name: Optional[str] = None
    # data packing (image SFT)
    enable_data_packing: bool = False
    max_packed_tokens: Optional[int] = None
    long_seq_threshold: Optional[int] = field(
        default=4096, metadata={"help": "Threshold to be a long single instance"}
    )
    max_instance_per_batch: Optional[int] = 4
    max_steps: Optional[int] = 2500


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data (local JSON)."}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Local dir the record image paths are resolved against."},
    )
    video_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Local dir the record video paths are resolved against."},
    )
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    # Pixel budget for the bbox crop / clue frames re-encoded as the alignment target.
    # Defaults give ~4..64 merged visual tokens per crop (factor 28 = patch 14 * merge 2).
    vtk_min_pixels: Optional[int] = field(default=4 * 28 * 28)
    vtk_max_pixels: Optional[int] = field(default=64 * 28 * 28)
    video_min_pixels: Optional[int] = field(default=16 * 28 * 28)
    video_max_pixels: Optional[int] = field(default=256 * 28 * 28)
    image_resized_width: Optional[int] = field(default=None)
    image_resized_height: Optional[int] = field(default=None)
    video_resized_width: Optional[int] = field(default=None)
    video_resized_height: Optional[int] = field(default=None)
    fps: float = 1.0
    random_seed: Optional[int] = field(default=None)
    area_threshold: Optional[float] = field(default=1.0)
    gen_tokens: Optional[int] = field(default=64)
    # Video: frames uniformly sampled from the clue temporal span to form the <VR> alignment
    # target (the video analogue of the image bbox crop).
    num_clue_frames: int = field(default=3)
    # Video: upper bound on frames decoded for the full-video input (the question context).
    max_frames: int = field(default=128)
