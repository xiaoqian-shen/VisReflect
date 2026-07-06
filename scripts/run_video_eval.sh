#!/bin/bash
# Run VisReflect (Qwen2.5-VL-VTK) VIDEO-benchmark evals locally with torchrun -- one job per
# task, one rank per GPU. Benchmarks are pulled from HuggingFace and results are written
# under OUTPUT_DIR.
# Video tasks: videomme, mlvu, mvbench. Each job SELF-STAGES its data: rank 0 downloads +
# unzips the video shards to $VISREFLECT_VIDEO_CACHE (slow first run -- MLVU ~280GB,
# Video-MME ~100GB, MVBench ~16GB; needs 100s of GB of local disk), then all ranks read it.
#
# fps, max_frames, and the per-frame/total PIXEL budgets are baked PER-BENCHMARK in
# evaluation/eval_config.py -- they are NOT env vars here.
#
# Usage:
#   MODEL_PATH=/path/to/ckpt ./scripts/run_video_eval.sh                     # default: mlvu
#   BENCHMARKS=videomme,mlvu,mvbench MODEL_PATH=/path/to/ckpt ./scripts/run_video_eval.sh
#   LIMIT=20 BENCHMARKS=mvbench MODEL_PATH=/path/to/ckpt ./scripts/run_video_eval.sh  # smoke test
#
# Env overrides: MODEL_PATH, BENCHMARKS, OUTPUT_DIR, STEPS, MAX_NEW_TOKENS, REPETITION_PENALTY,
#   NO_REPEAT_NGRAM_SIZE, CONSTRAIN_ANSWER, LIMIT, SAMPLE_SEED, GPUS_PER_NODE,
#   VISREFLECT_ATTN_IMPL, VISREFLECT_VIDEO_CACHE.
#
# MVBench is a gated HuggingFace dataset -- run `huggingface-cli login` first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a checkpoint dir or HuggingFace model id}"
BENCHMARKS="${BENCHMARKS:-mlvu}" # videomme,mlvu,mvbench
OUTPUT_DIR="${OUTPUT_DIR:-./visreflect_video_eval_results}"
STEPS="${STEPS:-4}" # >=1 VTK reflection (steps=N); 0 = no-reflection baseline
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
# Anti-degeneration for the answer after <EOR> (breaks "1 1 1 1" / repeated-CJK loops on
# uncertain videos); only used when CONSTRAIN_ANSWER=False.
REPETITION_PENALTY="${REPETITION_PENALTY:-1.3}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-3}"
# Strict MCQ decoding (default): after <EOR>, force the answer prefix and argmax the option
# letter -> always a valid letter. Set False to free-generate with the repetition knobs above.
CONSTRAIN_ANSWER="${CONSTRAIN_ANSWER:-True}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

IFS=',' read -ra TASKS <<<"${BENCHMARKS}"
echo "=== running ${#TASKS[@]} video task(s): ${BENCHMARKS} ==="
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
        --repetition_penalty "${REPETITION_PENALTY}"
        --no_repeat_ngram_size "${NO_REPEAT_NGRAM_SIZE}"
        --constrain_answer "${CONSTRAIN_ANSWER}"
    )
    if [ -n "${LIMIT:-}" ]; then
        EVAL_ARGS+=(--limit "${LIMIT}")
    fi
    if [ -n "${SAMPLE_SEED:-}" ]; then
        EVAL_ARGS+=(--sample_seed "${SAMPLE_SEED}")
    fi

    echo "--- ${task} (1x${GPUS_PER_NODE}) ---"
    torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" \
        -m evaluation.eval_config --task "${task}" \
        "${EVAL_ARGS[@]}"
done
