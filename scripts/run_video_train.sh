#!/bin/bash
# Run VisReflect (Qwen2.5-VL-VTK) VIDEO SFT (visual-reflection alignment) locally with
# torchrun -- one rank per GPU. Reads a local records JSON + a local video folder (no
# Manifold, no streaming) and writes checkpoints to OUTPUT_DIR. One video = one sequence.
#
# DATA_PATH: a local JSON list, each record
#   {"video": "<path rel to VIDEO_FOLDER or absolute>",
#    "question": "<question text>",
#    "answer": "<answer text>",
#    "temporal_span": [lo_sec, hi_sec]}   # or a list of [lo, hi] pairs (multiple clue spans)
# `num_clue_frames` frames uniformly sampled from the temporal span become the <VR> alignment
# target; the answer is trained as `<BOR><VR>..<EOR>The answer is: <answer>`.
#
# Usage:
#   MODEL_ID=/path/to/base_ckpt DATA_PATH=/path/video_records.json VIDEO_FOLDER=/path/videos \
#       ./scripts/run_video_train.sh
#
# Env overrides: MODEL_ID, DATA_PATH, VIDEO_FOLDER, OUTPUT_DIR, GPUS_PER_NODE, MAX_STEPS,
#   MAX_SEQ_LEN, NUM_CLUE_FRAMES, MAX_FRAMES, FPS, GRAD_ACCUM_STEPS, LEARNING_RATE,
#   LOSS_ALIGN_LAMBDA, FREEZE_VISION_TOWER, FREEZE_MERGER, FREEZE_LLM, DISABLE_FLASH_ATTN,
#   FSDP, RUN_NAME. Single GPU: GPUS_PER_NODE=1 FSDP="".
#
# Init MODEL_ID from the IMAGE VTK SFT checkpoint (continual image->video training); it already
# carries the <BOR>/<VR>/<EOR> tokens + vtk_*_id config, so no new tokens are added.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:?set MODEL_ID to a base/image-VTK checkpoint dir or HuggingFace model id}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to a local video records JSON}"
VIDEO_FOLDER="${VIDEO_FOLDER:?set VIDEO_FOLDER to the local video root}"
OUTPUT_DIR="${OUTPUT_DIR:-./visreflect_video_train_output}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-5000}"
# Per-video max sequence length (fetch_video spreads this budget across sampled frames).
MAX_SEQ_LEN="${MAX_SEQ_LEN:-20000}"
NUM_CLUE_FRAMES="${NUM_CLUE_FRAMES:-3}"
MAX_FRAMES="${MAX_FRAMES:-256}"
FPS="${FPS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
# Video default 0.0: the visual-reflection signal comes from the latent-CE two-pass
# (vtk_train_latent_ce), not the cosine align loss. Set >0 to also pull <VR> to the clue frames.
LOSS_ALIGN_LAMBDA="${LOSS_ALIGN_LAMBDA:-0.0}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-True}"
FREEZE_LLM="${FREEZE_LLM:-False}"
DISABLE_FLASH_ATTN="${DISABLE_FLASH_ATTN:-False}"
FSDP="${FSDP:-full_shard auto_wrap}"
RUN_NAME="${RUN_NAME:-visreflect-video}"

TRAIN_ARGS=(
    --model_id "${MODEL_ID}"
    --data_path "${DATA_PATH}"
    --video_folder "${VIDEO_FOLDER}"
    --output_dir "${OUTPUT_DIR}"
    --bf16 True
    --fp16 False
    --tf32 True
    --disable_flash_attn2 "${DISABLE_FLASH_ATTN}"
    # --- VTK / VisReflect alignment (clue frames sampled from the temporal span) ---
    --loss_align_lambda "${LOSS_ALIGN_LAMBDA}"
    --num_clue_frames "${NUM_CLUE_FRAMES}"
    --max_frames "${MAX_FRAMES}"
    --fps "${FPS}"
    --vtk_min_pixels "$((16 * 28 * 28))"
    --vtk_max_pixels "$((64 * 28 * 28))"
    --video_min_pixels "$((128 * 28 * 28))"
    --video_max_pixels "$((256 * 28 * 28))"
    # --- what to train ---
    --freeze_vision_tower "${FREEZE_VISION_TOWER}"
    --freeze_merger "${FREEZE_MERGER}"
    --freeze_llm "${FREEZE_LLM}"
    # per-video max sequence length (see fetch_video)
    --max_packed_tokens "${MAX_SEQ_LEN}"
    # --- optimization / schedule ---
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

echo "=== video training ${RUN_NAME} (1x${GPUS_PER_NODE}) ==="
echo "    model:  ${MODEL_ID}"
echo "    data:   ${DATA_PATH}  (videos: ${VIDEO_FOLDER})"
echo "    output: ${OUTPUT_DIR}"

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
    -m src.train.train_video \
    "${TRAIN_ARGS[@]}"
