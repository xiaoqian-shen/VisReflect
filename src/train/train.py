from src.dataset import (
    make_packed_supervised_data_module,
    make_supervised_data_module,
)
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.monkey_patch import replace_train_dataloader
from src.train.train_utils import (
    build_vtk_model_and_processor,
    safe_save_model_for_hf_trainer,
)
from src.trainer import QwenTrainer
from transformers import HfArgumentParser


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = build_vtk_model_and_processor(
        model_args, data_args, training_args
    )

    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1
        data_module, total_data_len = make_packed_supervised_data_module(
            model_id=model_args.model_id,
            processor=processor,
            data_args=data_args,
            training_args=training_args,
        )
        if not training_args.max_steps:
            training_args.max_steps = total_data_len // (
                training_args.gradient_accumulation_steps
                * training_args.world_size
                * training_args.per_device_train_batch_size
            )
        # Crucial or the packed data gets incorrectly re-sharded by the dataloader.
        replace_train_dataloader()
    else:
        data_module = make_supervised_data_module(
            model_id=model_args.model_id, processor=processor, data_args=data_args
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
