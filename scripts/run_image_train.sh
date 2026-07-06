#!/bin/bash
# Run VisReflect (Qwen2.5-VL-VTK) IMAGE SFT locally with torchrun -- one rank per GPU.
# Reads a local records JSON + a local image folder and writes checkpoints to OUTPUT_DIR.
# Hyperparameters mirror the original Visual-CoT SFT recipe.
#
# DATA_PATH accepts the raw Visual-CoT viscot_363k.json directly (converted on load), or the
# internal records JSON: a list of objects, each
#   {"image": "<path rel to IMAGE_FOLDER>" (or a list of paths),
#    "bboxes": [[x0, y0, x1, y1], ...],   # normalized xyxy; the visual-reflection crops
#    "conversations": [{"from": "human", "value": "<image>\n<question>"},
#                      {"from": "gpt",   "value": "The answer is: X"}]}
# (A meta-manifest list of {data_path, image_folder, ds_name} is also accepted.)
#
# Usage:
#   MODEL_ID=/path/to/base_ckpt DATA_PATH=/path/records.json IMAGE_FOLDER=/path/images \
#       ./scripts/run_image_train.sh
#
# Env overrides: MODEL_ID, DATA_PATH, IMAGE_FOLDER, OUTPUT_DIR, GPUS_PER_NODE, MAX_STEPS,
#   MAX_PACKED_TOKENS, MAX_INSTANCE_PER_BATCH, GRAD_ACCUM_STEPS, LEARNING_RATE,
#   FREEZE_VISION_TOWER, FREEZE_MERGER, FREEZE_LLM, DISABLE_FLASH_ATTN, FSDP, RUN_NAME.
# Single GPU: pass GPUS_PER_NODE=1 FSDP="" (FSDP full_shard is for multi-GPU).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:?set MODEL_ID to a base checkpoint dir or HuggingFace model id}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to a local records JSON}"
IMAGE_FOLDER="${IMAGE_FOLDER:?set IMAGE_FOLDER to the local image root}"
OUTPUT_DIR="${OUTPUT_DIR:-./visreflect_train_output}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-5000}"
MAX_PACKED_TOKENS="${MAX_PACKED_TOKENS:-16384}"
MAX_INSTANCE_PER_BATCH="${MAX_INSTANCE_PER_BATCH:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-True}"
FREEZE_LLM="${FREEZE_LLM:-False}"
DISABLE_FLASH_ATTN="${DISABLE_FLASH_ATTN:-False}"
FSDP="${FSDP:-full_shard auto_wrap}"
RUN_NAME="${RUN_NAME:-visreflect-image}"

TRAIN_ARGS=(
    --model_id "${MODEL_ID}"
    --data_path "${DATA_PATH}"
    --image_folder "${IMAGE_FOLDER}"
    --output_dir "${OUTPUT_DIR}"
    --bf16 True
    --fp16 False
    --tf32 True
    --disable_flash_attn2 "${DISABLE_FLASH_ATTN}"
    # --- VTK / VisReflect alignment ---
    --loss_align_lambda 0.1
    --area_threshold 1.0
    --vtk_min_pixels "$((16 * 28 * 28))"
    --vtk_max_pixels "$((128 * 28 * 28))"
    --image_min_pixels "$((128 * 28 * 28))"
    --image_max_pixels "$((4096 * 28 * 28))"
    # --- what to train ---
    --freeze_vision_tower "${FREEZE_VISION_TOWER}"
    --freeze_merger "${FREEZE_MERGER}"
    --freeze_llm "${FREEZE_LLM}"
    # --- sequence packing ---
    --enable_data_packing True
    --max_packed_tokens "${MAX_PACKED_TOKENS}"
    --long_seq_threshold 4096
    --max_instance_per_batch "${MAX_INSTANCE_PER_BATCH}"
    # --- optimization / schedule (max_steps required: packed data is an IterableDataset) ---
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --random_seed 42
    # --- logging / checkpointing (local) ---
    --logging_steps 1
    --logging_first_step True
    --save_strategy steps
    --save_steps 500
    --save_total_limit 2
    --report_to tensorboard
    --remove_unused_columns False
    --run_name "${RUN_NAME}"
)
if [ -n "${FSDP}" ]; then
    TRAIN_ARGS+=(--fsdp "${FSDP}")
fi

echo "=== training ${RUN_NAME} (1x${GPUS_PER_NODE}) ==="
echo "    model:  ${MODEL_ID}"
echo "    data:   ${DATA_PATH}  (images: ${IMAGE_FOLDER})"
echo "    output: ${OUTPUT_DIR}"

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
    -m src.train.train \
    "${TRAIN_ARGS[@]}"
