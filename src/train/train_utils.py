import logging

import torch
import transformers
from transformers import AutoConfig, AutoProcessor

from src.constants import VTK_END_TOKEN, VTK_START_TOKEN, VTK_TOKEN
from src.model.qwen2_5_model import Qwen2_5_VTK
from src.train.monkey_patch import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_5_vl_patch_emb,
)

logger: logging.Logger = logging.getLogger(__name__)


def mean_init_new_token_embeddings(model, num_new_tokens: int) -> None:
    """Initialize the last ``num_new_tokens`` rows of the input/output embeddings to the mean of
    the pre-existing rows, instead of the default random init. Keeps the freshly added
    <BOR>/<VR>/<EOR> tokens near-neutral so inserting the reflection block perturbs the base
    model's next-token distribution -- and thus the answer -- as little as possible. No-op when
    no rows were added (e.g. continuing from a checkpoint that already carries the tokens), so it
    never clobbers already-trained token embeddings."""
    if num_new_tokens <= 0:
        return
    with torch.no_grad():
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            if emb is None:
                continue
            weight = emb.weight
            weight[-num_new_tokens:] = weight[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )


def _set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def _configure_trainable_params(model, training_args, compute_dtype):
    """Freeze/unfreeze per the --freeze_* flags. Order matters: the LLM pass covers all of
    ``model.model`` (which includes the vision tower + merger), then the vision-tower and merger
    passes override those two. The vision tower is also moved to the compute dtype/device."""
    _set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    _set_requires_grad(model.model.parameters(), not training_args.freeze_llm)

    vision_tower = model.model.visual
    vision_tower.to(dtype=compute_dtype, device=training_args.device)
    _set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    _set_requires_grad(vision_tower.merger.parameters(), not training_args.freeze_merger)


def build_vtk_model_and_processor(
    model_args, data_args, training_args, set_latent_ce: bool = False
):
    """Load the Qwen2.5-VL VTK model + processor for training, shared by image and video.

    Patches the mixed-modality forward and the fp32 patch-embed, configures which params train,
    adds the <BOR>/<VR>/<EOR> tokens (mean-initialized), and pins the embedding dtype. Detects
    the model via ``config.model_type`` (a checkpoint dir need not have "Qwen2.5" in its name).
    ``set_latent_ce=True`` (video) turns on the two-pass latent-CE forward; image leaves the
    model config untouched (single pass)."""
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model_pth = training_args.checkpoint_name or model_args.model_id
    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    if "qwen2_5_vl" not in getattr(config, "model_type", "").lower():
        raise ValueError(
            "Unsupported model type. At this moment, we only support Qwen2.5LM-based "
            "Qwen2.5VL series."
        )

    replace_qwen2_5_with_mixed_modality_forward()
    model = Qwen2_5_VTK.from_pretrained(
        model_pth,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2"
        if not training_args.disable_flash_attn2
        else "sdpa",
    )
    replace_qwen_2_5_vl_patch_emb()  # fp32 patch-emb; edge-case numerical stability

    model.config.use_cache = False
    _configure_trainable_params(model, training_args, compute_dtype)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        # FSDP requires non-reentrant checkpointing; DeepSpeed / single-GPU use reentrant.
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": not bool(training_args.fsdp)
        }

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )
    for tok in (VTK_START_TOKEN, VTK_TOKEN, VTK_END_TOKEN):
        processor.tokenizer.add_tokens(tok, special_tokens=True)
    model.config.vtk_id = processor.tokenizer.convert_tokens_to_ids(VTK_TOKEN)
    model.config.vtk_start_id = processor.tokenizer.convert_tokens_to_ids(VTK_START_TOKEN)
    model.config.vtk_end_id = processor.tokenizer.convert_tokens_to_ids(VTK_END_TOKEN)
    if set_latent_ce:
        # Option A (video only): compute CE on the model's own latent reflection instead of the
        # answer-leaking clue frames. See monkey_patch.qwen2_5_mixed_modality_forward.
        model.config.vtk_train_latent_ce = training_args.vtk_train_latent_ce

    # there are some dummy tokens in newer hf version. Mean-init only the rows the resize
    # actually appended (0 when the checkpoint already carries the VTK tokens).
    old_vocab = model.get_input_embeddings().weight.shape[0]
    if model.config.text_config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    mean_init_new_token_embeddings(
        model, max(0, model.get_input_embeddings().weight.shape[0] - old_vocab)
    )
    # resize_token_embeddings makes a fresh fp32 lm_head/embedding on an untied bf16 model; pin
    # both back to the compute dtype or eval (torch_dtype="auto") crashes at lm_head on a dtype
    # mismatch.
    model.get_input_embeddings().to(compute_dtype)
    model.get_output_embeddings().to(compute_dtype)
    return model, processor


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collect the state dict and dump it to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)
