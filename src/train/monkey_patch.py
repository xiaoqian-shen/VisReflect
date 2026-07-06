"""Runtime monkey-patches for Qwen2.5-VL VTK training/eval, applied by the entry points:

  * replace_qwen2_5_with_mixed_modality_forward -- the VTK mixed-modality forward (re-encode
    the crop / clue frames into the <VR> slots, loss_ce + loss_align, optional latent-CE
    second pass for video).
  * replace_qwen_2_5_vl_patch_emb -- fp32 vision patch-embed for numerical stability.
  * replace_train_dataloader -- a plain (non-IterableDatasetShard) train DataLoader for the
    packed / iterable datasets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import datasets
import torch
import torch.distributed as dist
import transformers
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.trainer import is_datasets_available, seed_worker
from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
from src.constants import IGNORE_INDEX


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def replace_qwen2_5_with_mixed_modality_forward():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_align: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None

    # Pre-lm_head hidden states the CE logits are read off (the latent-reflection pass when
    # vtk_train_latent_ce is on).
    ce_hidden_states: Optional[torch.FloatTensor] = None


def compute_vtk_align_loss(
    model, hidden_states, vtk_embeds, batch_indices, seq_positions
):
    """Cosine-alignment loss between the LLM hidden state right before each
    <|vtk|> token and the freshly re-encoded crop embedding it should predict."""
    has_local_vtk = vtk_embeds is not None and vtk_embeds.shape[0] > 0
    has_embeds = torch.tensor(1.0 if has_local_vtk else 0.0).to(model.model.device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(has_embeds, op=dist.ReduceOp.SUM)

    if has_embeds <= 0:
        return torch.tensor(0.0).to(model.model.device)
    if not has_local_vtk:
        # rank has no crops this step; keep grad graph connected with a zero
        return hidden_states.sum() * 0.0

    selected = hidden_states[batch_indices, seq_positions - 1].to(torch.float32)
    selected = selected / torch.norm(selected, dim=-1, keepdim=True)
    vtk_n = vtk_embeds.to(torch.float32)
    vtk_n = vtk_n / torch.norm(vtk_n, dim=-1, keepdim=True)
    # Per-position (diagonal) cosine: align each <VR>'s preceding hidden state to ITS own
    # crop/frame embedding (the one scattered into that position above), NOT every hidden
    # to every embedding. The old `(vtk_n @ selected.T).mean()` averaged the full [N,N]
    # pairwise matrix, which for video's ~hundreds of diverse clue-frame tokens collapses
    # all reflection hidden states to the mean embedding (image escaped it: few, near-
    # identical per-crop tokens).
    return 1 - (vtk_n * selected).sum(dim=-1).mean()


def _latent_reflection_ce_hidden(
    model,
    lm_kwargs,
    hidden_states,
    inputs_embeds,
    input_ids,
    labels,
):
    """Option A: return hidden states for the CE loss computed on the LATENT reflection (the one
    eval uses), instead of whatever is scattered into the <VR> slots in Pass 1.

    At eval the <VR> slots carry the model's OWN pre-<VR> hidden state (latent reflection); at
    train Pass 1 they carry either the real clue-frame embeddings (grounded video) or just the
    plain <VR> token embedding (mixed / no-align: fixed <VR> block, no clue frames). Either way,
    training CE on Pass 1 mismatches eval and produces garbage after <EOR>. So re-run the LM with
    the <VR> slots fed the DETACHED pre-<VR> hidden state (== eval) and take CE from that pass;
    Pass 1 stays the loss_align target (a no-op when there are no clue frames). detach: aligning
    the latent to the clue frame is loss_align's job -- CE must not smuggle the answer into it.

    Gated on ``model.config.vtk_train_latent_ce`` (set by the video/mixed train entries only),
    ``labels`` (so generation is untouched), and the presence of <VR> tokens; returns Pass-1
    ``hidden_states`` unchanged otherwise -- image training and all eval keep the single pass.
    """
    if (
        labels is None
        or input_ids is None
        or not getattr(model.config, "vtk_train_latent_ce", False)
    ):
        return hidden_states
    # Recompute the <VR> slots here (rather than reuse the scatter block's) so this works
    # whether or not there were clue-frame crops -- mixed/no-align training has <VR> but no
    # vtk_tokens, so the forward's batch_indices would be None.
    batch_indices, seq_positions = torch.nonzero(
        input_ids == model.config.vtk_id, as_tuple=True
    )
    if batch_indices.numel() == 0:
        return hidden_states
    latents = hidden_states[batch_indices, seq_positions - 1].detach()
    inputs_embeds_ce = inputs_embeds.clone()
    inputs_embeds_ce[batch_indices, seq_positions] = latents.to(inputs_embeds.dtype)
    return model.model.language_model(inputs_embeds=inputs_embeds_ce, **lm_kwargs)[0]


def qwen2_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    vtk_tokens: Optional[torch.Tensor] = None,
    vtk_grid: Optional[List[torch.Tensor]] = None,
    mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    # Latent reflection: for the batch rows currently inside the VTK window, overwrite
    # the last-position embedding with the previous step's hidden state. `mode_switch`
    # is a per-row bool tensor during generation and `None` during training -- guard
    # with `.any()` rather than truthiness so batched (batch_size > 1) generation does
    # not trip "Boolean value of Tensor ... is ambiguous".
    in_reflection = mode_switch is not None and bool(mode_switch.any())
    if last_position_hidden_state is not None and mode_switch is not None:
        inputs_embeds[mode_switch, -1, :] = last_position_hidden_state[mode_switch]

    if not in_reflection and (pixel_values is None and pixel_values_videos is None):
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)

        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id

        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0)
        if input_ids is None:
            video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.video_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            video_mask = video_mask.all(-1)
        else:
            video_mask = input_ids == self.config.video_token_id
        n_video_tokens = (video_mask).sum()
        video_mask_unsqueeze = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        n_video_features = video_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_unsqueeze, video_embeds)

    # Re-encode the bbox crop(s) freshly through the ViT and scatter the crop
    # embeddings into the <|vtk|> placeholder positions. vtk_tokens holds crop
    # pixel_values; vtk_grid holds their grid_thw.
    vtk_embeds = None
    batch_indices = None
    seq_positions = None
    if vtk_tokens is not None and vtk_tokens.numel() > 0:
        vtk_mask = input_ids == self.config.vtk_id
        batch_indices, seq_positions = torch.nonzero(vtk_mask, as_tuple=True)
        vtk_embeds = self.get_image_features(vtk_tokens, vtk_grid)
        vtk_embeds = torch.cat(vtk_embeds, dim=0)
        vtk_embeds = vtk_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds[batch_indices, seq_positions] = vtk_embeds
    else:
        # No crops this step: still run the crop encoder on a dummy crop (output multiplied
        # by 0, so it affects nothing) to keep the forward's module-call sequence identical
        # to the crop path. Mirrors the no-image dummy above.
        dummy_pixel = (
            torch.zeros(784, 1176)
            .type(self.model.visual.dtype)
            .to(self.model.visual.device)
        )
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        dummy_embeds = self.get_image_features(dummy_pixel, dummy_grid)
        dummy_embeds = torch.cat(dummy_embeds, dim=0)
        inputs_embeds = inputs_embeds + dummy_embeds.mean() * 0

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (
            prefill_compiled_stage or prefill_noncompiled_stage
        ) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(
                    inputs_embeds.device
                )
            else:
                delta = torch.zeros(
                    (batch_size, seq_length), device=inputs_embeds.device
                )
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    # Shared LM-forward kwargs (everything except inputs_embeds) so the CE latent-reflection
    # pass can re-run the language model with identical position ids / attention / cache.
    lm_kwargs = dict(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    # Pass 1: <VR> slots hold the real clue-frame embeddings (scattered above). This is the
    # loss_align source (hidden state right before each <VR> is pulled toward its clue frame).
    outputs = self.model.language_model(inputs_embeds=inputs_embeds, **lm_kwargs)
    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]

    # CE runs on the latent reflection (video Option A) when enabled, else on Pass 1.
    ce_hidden_states = _latent_reflection_ce_hidden(
        self,
        lm_kwargs,
        hidden_states,
        inputs_embeds,
        input_ids,
        labels,
    )

    logits = self.lm_head(ce_hidden_states)

    loss_ce = None
    loss_align = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.masked_fill(
            shift_labels == self.config.vtk_id, IGNORE_INDEX
        )

        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        loss_align = compute_vtk_align_loss(
            self, hidden_states, vtk_embeds, batch_indices, seq_positions
        )

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_align=loss_align,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=last_position_hidden_state,
        ce_hidden_states=ce_hidden_states,
    )


# --------------------------------------------------------------------------- #
# fp32 vision patch-embed (avoids an edge-case numerical-stability issue)
# --------------------------------------------------------------------------- #
def Qwen2_5_VisionPatchEmbedFp32Forward(
    self, hidden_states: torch.Tensor
) -> torch.Tensor:
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )
    with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
        hidden_states = self.proj(hidden_states)
    hidden_states = hidden_states.view(-1, self.embed_dim).to(target_dtype)
    return hidden_states


def replace_qwen_2_5_vl_patch_emb():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbedFp32Forward


# --------------------------------------------------------------------------- #
# Plain train DataLoader for packed / iterable datasets (adapted from InternVL:
# https://github.com/OpenGVLab/InternVL). Bypasses HF's IterableDatasetShard so a
# packed/iterable dataset that shards internally is not re-read once per rank.
# --------------------------------------------------------------------------- #
def get_train_dataloader(self) -> DataLoader:
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(
            train_dataset, description="training"
        )
    else:
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    dataloader = DataLoader(train_dataset, **dataloader_params)
    # enable_data_packing feeds pre-collated batches; the accelerator prepare wrapping is
    # only needed for the standard (non-packed) map-style path.
    if self.args.enable_data_packing:
        return dataloader
    return self.accelerator.prepare(dataloader)


def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
