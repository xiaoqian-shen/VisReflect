from src.dataset import make_supervised_data_module_video
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.train_utils import (
    build_vtk_model_and_processor,
    safe_save_model_for_hf_trainer,
)
from src.trainer import QwenTrainer
from transformers import HfArgumentParser


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set_latent_ce=True: video trains CE on the model's own latent reflection (Option A).
    model, processor = build_vtk_model_and_processor(
        model_args, data_args, training_args, set_latent_ce=True
    )

    # One video = one sequence (no token-packing); the collator batches
    # per_device_train_batch_size videos, each padded. max_packed_tokens is the per-video
    # max sequence length (fetch_video spreads the budget across frames).
    data_module = make_supervised_data_module_video(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        training_args=training_args,
    )

    trainer = QwenTrainer(
        model=model, processing_class=processor, args=training_args, **data_module
    )
    trainer.train(resume_from_checkpoint=bool(training_args.checkpoint_name))
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
