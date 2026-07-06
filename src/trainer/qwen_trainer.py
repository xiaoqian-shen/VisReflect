import os

import torch.nn as nn
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    ExportableState,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    logger,
    PREFIX_CHECKPOINT_DIR,
    SaveStrategy,
    TRAINER_STATE_NAME,
)


class QwenTrainer(Trainer):
    def __init__(self, *args, temp_folder=None, oci_handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Running sums of the extra metrics (loss components, token counts). compute_loss
        # accumulates into these every micro-batch; log() flushes their average once per
        # logging step -- logging in compute_loss directly writes one TensorBoard point
        # per micro-batch, i.e. several points at the same global_step under grad accum.
        self._metric_sums = {}
        self._metric_count = 0
        # if online checkpointing
        if oci_handler:
            self.oci_handler = oci_handler
            self.temp_folder = temp_folder  # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # modified to support online checkpointing
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        # output_dir is the local path forcheckpoint
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if (
            self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
            and self.state.best_global_step
        ):
            best_checkpoint_folder = (
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            )
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation (use_mtime=False);
            # mtime is unreliable on some fuse fs in cloud environments. The inherited
            # method reads self.args.save_total_limit and preserves the best checkpoint.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        metrics = {}
        if self.args.enable_data_packing:
            batch_size = inputs["input_ids"].size(0)
            total_tokens = inputs["input_ids"].size(0) * inputs["input_ids"].size(1)
            vtk_grid = inputs.get("vtk_grid", None)
            if vtk_grid is not None and vtk_grid.numel() > 0:
                num_crops = vtk_grid.size(0)
                # merged vtk tokens per crop = prod(grid_thw) / merge_size**2 (=/4)
                total_vtk = int(vtk_grid.prod(dim=1).sum().item()) // 4
                gen_tokens = total_vtk / num_crops
            else:
                gen_tokens = 0
            metrics.update(
                {
                    "batch_size": batch_size,
                    "tokens_per_device": total_tokens,
                    "tokens_gen": round(gen_tokens),
                }
            )

        outputs = model(**inputs)
        loss_ce = outputs.loss_ce
        loss_align = outputs.loss_align

        loss = (
            loss_ce + self.args.loss_align_lambda * loss_align
            if self.args.loss_align_lambda > 0
            else loss_ce
        )

        metrics.update(
            {
                "loss_total": loss.detach().item(),
                "loss_ce": loss_ce.detach().item(),
                "loss_align": loss_align.detach().item(),
            }
        )
        # Accumulate; the average is emitted once per logging step in log() (calling
        # self.log here would write one point per micro-batch -> several values at the
        # same global_step when gradient_accumulation_steps > 1).
        for k, v in metrics.items():
            self._metric_sums[k] = self._metric_sums.get(k, 0.0) + v
        self._metric_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Fold the per-step average of the accumulated extra metrics into HF's single
        # per-logging-step log call -> one value per global_step (written on the main
        # process by the TensorBoard callback).
        if self._metric_count > 0:
            n = self._metric_count
            for k, total in self._metric_sums.items():
                logs[k] = total / n
            self._metric_sums = {}
            self._metric_count = 0
        return super().log(logs, *args, **kwargs)
