import inspect
import os
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerateOutput,
    logger,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module


class Qwen2_5_VTK(Qwen2_5_VLForConditionalGeneration):
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        steps: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if steps is None and hasattr(generation_config, "steps"):
            steps = generation_config.steps

        # 1. Handle `generation_config` and kwargs that might update it.
        tokenizer = kwargs.pop(
            "tokenizer", None
        )  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop(
            "assistant_tokenizer", None
        )  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (
                is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
            ) and dist.get_world_size() > 1

        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys()
        )
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(
            generation_config, kwargs_has_attention_mask, device=device
        )

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(
                    inputs_tensor[:, -1] == generation_config._pad_token_tensor
                )
                > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if (
            not kwargs_has_attention_mask
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs["attention_mask"] = (
                self._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config, model_kwargs
                )
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if (
                model_input_name == "input_ids"
                and len(model_kwargs["attention_mask"].shape) > 2
            ):
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = (
                inputs_tensor
                if model_input_name == "input_ids"
                else model_kwargs.pop("input_ids")
            )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]

        generation_mode = generation_config.get_generation_mode(assistant_model)
        # transformers changed this helper's signature across versions: some releases
        # require a `device` arg and name the 3rd param `assistant_model`, others take
        # `generation_mode` there and no device. Adapt to whatever the installed
        # transformers exposes so the VTK decode loop below stays version-robust.
        _cache_sig = inspect.signature(self._prepare_cache_for_generation)
        _cache_params = list(_cache_sig.parameters)
        _third = assistant_model if _cache_params[2] == "assistant_model" else generation_mode
        _cache_args = [generation_config, model_kwargs, _third, batch_size, max_cache_length]
        if "device" in _cache_sig.parameters:
            self._prepare_cache_for_generation(*_cache_args, device=inputs_tensor.device)
        else:
            self._prepare_cache_for_generation(*_cache_args)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        result = self._decoding_by_steps(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            steps=steps,
            **model_kwargs,
        )

        return result

    def _decoding_by_steps(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        steps: List[int],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        try:
            model_kwargs = self._get_initial_cache_position(
                cur_len, input_ids.device, model_kwargs
            )
        except TypeError:
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = (
                model_kwargs["past_key_values"].is_compileable
                and self._supports_static_cache
            )
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda"
                or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(
                input_ids, generation_config, **model_kwargs
            )
            is_prefill = False
        else:
            is_prefill = True

        mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        last_position_hidden_state = None

        steps_orig = torch.tensor(
            steps, dtype=torch.int, device=input_ids.device
        )  # original quota
        remaining_steps = steps_orig.clone()
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            model_inputs.update({"mode_switch": mode_switch})
            model_inputs.update(
                {"last_position_hidden_state": last_position_hidden_state}
            )
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            last_tokens = input_ids[:, -1]
            start_switch = (last_tokens == self.config.vtk_start_id).to(
                device=input_ids.device
            )
            new_mode_switch = mode_switch | start_switch

            just_entered = (~mode_switch) & new_mode_switch
            remaining_steps = torch.where(just_entered, steps_orig, remaining_steps)
            remaining_steps = remaining_steps - mode_switch.long()
            mode_switch = new_mode_switch & (remaining_steps > 0)
            # Reflection bookkeeping: while inside the window force the placeholder <VR> token
            # (its input embedding is overwritten by the previous hidden state in the forward
            # -- latent reflection), and emit <EOR> the step the quota runs out so the model
            # sees a real end-of-reflection token before decoding the answer. Without this the
            # window carries argmax garbage and the model never gets the <EOR> signal ->
            # degenerate output.
            next_tokens = torch.where(
                mode_switch,
                torch.full_like(next_tokens, self.config.vtk_id),
                next_tokens,
            )
            eor_switch = new_mode_switch & (~mode_switch)
            next_tokens = torch.where(
                eor_switch,
                torch.full_like(next_tokens, self.config.vtk_end_id),
                next_tokens,
            )
            last_position_hidden_state = outputs.last_position_hidden_state
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = mode_switch | (
                unfinished_sequences & ~stopping_criteria(input_ids, scores)
            )

            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
