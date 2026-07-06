#!/bin/bash
# Run VisReflect (Qwen2.5-VL-VTK) IMAGE-benchmark evals locally with torchrun -- one job per
# task, one rank per GPU. Benchmarks are pulled from HuggingFace and results are written
# under OUTPUT_DIR.
# Image tasks: vstar, hrbench4k, hrbench8k, blink. For videomme/mlvu/mvbench use
# scripts/run_video_eval.sh (their frame/pixel/decoding flags differ).
#
# Usage:
#   MODEL_PATH=/path/to/ckpt ./scripts/run_image_eval.sh                  # all image tasks
#   BENCHMARKS=vstar,hrbench8k MODEL_PATH=/path/to/ckpt ./scripts/run_image_eval.sh
#   LIMIT=20 BENCHMARKS=vstar MODEL_PATH=/path/to/ckpt ./scripts/run_image_eval.sh  # smoke test
#
# Env overrides: MODEL_PATH, BENCHMARKS, OUTPUT_DIR, STEPS, MAX_NEW_TOKENS, GPUS_PER_NODE,
#   LIMIT, VISREFLECT_ATTN_IMPL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a checkpoint dir or HuggingFace model id}"
BENCHMARKS="${BENCHMARKS:-vstar,hrbench4k,hrbench8k,blink}" # comma-separated tasks
OUTPUT_DIR="${OUTPUT_DIR:-./visreflect_eval_results}"
STEPS="${STEPS:-1}"           # >=1 VTK reflection (steps=N); 0 = no-reflection baseline
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

IFS=',' read -ra TASKS <<<"${BENCHMARKS}"
echo "=== running ${#TASKS[@]} task(s): ${BENCHMARKS} ==="
echo "    model:     ${MODEL_PATH}  (steps=${STEPS})"
echo "    results -> ${OUTPUT_DIR}"

for raw in "${TASKS[@]}"; do
    task="$(echo "${raw}" | xargs)" # trim whitespace
    [ -z "${task}" ] && continue

    EVAL_ARGS=(
        --model_path "${MODEL_PATH}"
        --output_dir "${OUTPUT_DIR}"
        --steps "${STEPS}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
    )
    if [ -n "${LIMIT:-}" ]; then
        EVAL_ARGS+=(--limit "${LIMIT}")
    fi

    echo "--- ${task} (1x${GPUS_PER_NODE}) ---"
    torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
        -m evaluation.eval_config --task "${task}" \
        "${EVAL_ARGS[@]}"
done
