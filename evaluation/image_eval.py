"""Shared machinery for the VisReflect image-benchmark evals.

The per-benchmark HuggingFace repo id and vision-token budget live in the ``IMAGE_TASKS``
registry in ``eval_config.py``, which selects a benchmark via ``--task`` and calls
``run_eval`` here. This module holds everything common: model loading, inference, scoring,
the dataset loaders, and the single- vs multi-GPU dispatch.

Benchmarks are read straight from their official HuggingFace repos (``--dataset``, e.g.
``craigwu/vstar_bench``); an existing local directory is also accepted. Results are written
to the local ``--output_dir``.

``run_eval`` evaluates ONE benchmark and dispatches on ``WORLD_SIZE``: launched under
torchrun it runs the rank-aware data-parallel path (one rank per GPU); otherwise a plain
single process (optionally self-spawning across GPUs via ``--num_gpus`` on a single host).
"""

import argparse
import base64
import io
import json
import os
import re
import string
from datetime import timedelta
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import get_dataset_config_names, load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from evaluation.hf_data import resolve_file, resolve_model, resolve_repo
from src.constants import (
    VTK_END_TOKEN,
    VTK_START_TOKEN,
    VTK_TOKEN,
)
from src.model.qwen2_5_model import Qwen2_5_VTK
from src.train.monkey_patch import replace_qwen2_5_with_mixed_modality_forward


MAX_NEW_TOKENS = 32
# Per-image vision-token budget (pixels; 1 merged token = 28*28 px). None -> let
# qwen_vl_utils use its defaults (max 16384*28*28 = 16384 tokens). Per-benchmark entry
# modules set the argparse defaults; run_eval copies them into these globals.
IMAGE_MIN_PIXELS = None
IMAGE_MAX_PIXELS = None


def derive_run_name(chkpt_pth: str) -> str:
    """Short name for the checkpoint, used as the results subdir. Basename of a local dir
    or the ``name`` half of a ``org/name`` HuggingFace model id."""
    return chkpt_pth.rstrip("/").split("/")[-1]


def get_task_instruction(bench_name):
    return "\nOutput the letter of the correct answer only."


def _image_content(img):
    """Image content dict for a chat message. Attach the per-image pixel budget when set
    -- process_vision_info reads `min_pixels`/`max_pixels` keys to smart-resize the image
    (same mechanism training uses in data_utils)."""
    content = {"type": "image", "image": img}
    if IMAGE_MIN_PIXELS is not None:
        content["min_pixels"] = IMAGE_MIN_PIXELS
    if IMAGE_MAX_PIXELS is not None:
        content["max_pixels"] = IMAGE_MAX_PIXELS
    return content


def create_messages(img_path, question):
    if not isinstance(img_path, list):
        vision_content = [_image_content(img_path)]
    else:
        vision_content = [_image_content(ip) for ip in img_path]
    vision_content.append({"type": "text", "text": question})
    return [{"role": "user", "content": vision_content}]


def load_model_and_processor(chkpt_pth, steps, device_map="auto"):
    """``device_map="auto"`` shards one replica across all visible GPUs (single-process).
    Under torchrun pass an explicit single-device map (e.g. ``{"": local_rank}``) so each
    rank holds a full replica on its own card.

    The attention backend defaults to ``flash_attention_2`` (what the checkpoints were
    validated with); set ``VISREFLECT_ATTN_IMPL=sdpa`` to run without a flash-attn build."""
    attn_impl = os.environ.get("VISREFLECT_ATTN_IMPL", "flash_attention_2")
    config = AutoConfig.from_pretrained(chkpt_pth)
    if steps == 0:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            chkpt_pth,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
    else:
        replace_qwen2_5_with_mixed_modality_forward()
        model = Qwen2_5_VTK.from_pretrained(
            chkpt_pth,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device_map,
        )

    # Raise the image processor's own pixel cap to match the requested budget. Without
    # this, process_vision_info resizes per the message-level max_pixels but the
    # processor then RE-resizes down to its config cap (e.g. 12845056) -- so a larger
    # message budget would be silently clamped. Mirrors ZwZ's
    # AutoProcessor.from_pretrained(..., max_pixels=16777216).
    proc_kwargs = {}
    if IMAGE_MIN_PIXELS is not None:
        proc_kwargs["min_pixels"] = IMAGE_MIN_PIXELS
    if IMAGE_MAX_PIXELS is not None:
        proc_kwargs["max_pixels"] = IMAGE_MAX_PIXELS
    processor = AutoProcessor.from_pretrained(chkpt_pth, **proc_kwargs)
    # Decoder-only batched generation requires left-padding; harmless for batch_size=1.
    processor.tokenizer.padding_side = "left"

    return model, processor


def _vtk_suppress_ids(model, processor):
    """Token ids of the reflection markers (<BOR>/<VR>/<EOR>). Read from the checkpoint
    config (set at train time), falling back to a tokenizer lookup."""
    cfg = model.config
    ids = [
        getattr(cfg, name, None) for name in ("vtk_start_id", "vtk_id", "vtk_end_id")
    ]
    ids = [i for i in ids if isinstance(i, int)]
    if ids:
        return ids
    unk = processor.tokenizer.unk_token_id
    out = []
    for tok in (VTK_START_TOKEN, VTK_TOKEN, VTK_END_TOKEN):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != unk:
            out.append(tid)
    return out


def run_inference(model, processor, images, texts, steps):
    """Batched inference. ``images`` / ``texts`` are parallel lists (one entry per
    sample; an entry may itself be a list of images for multi-image benchmarks like
    BLINK). Returns one decoded string per sample.

    Inputs are left-padded (set in ``load_model_and_processor``) so the trimmed prompt
    length is identical across the batch, which is what decoder-only batched generation
    requires for correct results."""
    messages_list = [create_messages(img, txt) for img, txt in zip(images, texts)]
    texts_formatted = [
        processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for m in messages_list
    ]

    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=texts_formatted,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        if steps >= 1:
            generated_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, steps=[steps]
            )
        else:
            # Baseline (steps=0): ban the reflection markers so the fine-tuned model
            # can't emit <BOR>/<VR>/<EOR> and decodes the answer directly.
            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
            suppress = _vtk_suppress_ids(model, processor)
            if suppress:
                gen_kwargs["suppress_tokens"] = suppress
            generated_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return output_text


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


"""Answer parsing + scoring"""

# Tried in order against the cleaned prediction; first match wins. Letters are the
# multiple-choice options (A, B, C, ...). Ordered most-explicit to least so that a
# phrasing like "The answer is C" is not pre-empted by a stray earlier capital.
_ANSWER_PATTERNS = [
    r"answer\s*is\s*[:\-]?\s*\(?([A-Z])\)?",
    r"answer\s*[:\-]\s*\(?([A-Z])\)?",
    r"\(([A-Z])\)",
    r"^\s*([A-Z])\s*[\.\):,]",
    r"^\s*([A-Z])\s*$",
]


def parse_predicted_letter(prediction: str) -> str | None:
    """Extract the chosen option letter from a raw generation.

    The model decodes with special tokens kept (e.g. ``"B<|im_end|>"``), and may also
    wrap the letter in prose ("The answer is (B).") or prefix VTK reflection markers
    (``<BOR><VR>...<EOR>``). Strip every angle-bracket token, then try a series of
    patterns from most to least explicit, falling back to the first standalone capital
    letter. Returns ``None`` when no letter can be recovered."""
    text = re.sub(r"<[^>]*>", " ", prediction).strip()
    if not text:
        return None
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-Z])\b", text)
    return m.group(1).upper() if m else None


def normalize_label(label: Any) -> str:
    """Coerce a gold label into a bare option letter (handles ``"B"``, ``"(B)"``)."""
    parsed = parse_predicted_letter(str(label))
    return parsed if parsed else str(label).strip().upper()


def score_results(results):
    """Compute overall + per-category accuracy from gathered predictions."""
    total, correct = 0, 0
    res_by_category: dict[str, dict[str, int]] = {}
    for res in results:
        pred = (
            parse_predicted_letter(res["prediction"][0]) if res["prediction"] else None
        )
        gold = normalize_label(res["label"])
        cat = res.get("category", "overall")
        res_by_category.setdefault(cat, {"total": 0, "correct": 0})
        is_correct = pred is not None and pred == gold
        correct += int(is_correct)
        res_by_category[cat]["correct"] += int(is_correct)
        total += 1
        res_by_category[cat]["total"] += 1
    return correct, total, res_by_category


def _print_accuracy(steps, correct, total, res_by_category=None):
    pct = correct / total * 100 if total else 0.0
    print(f"Steps: {steps} Overall - Accuracy: {correct}/{total} = {pct:.2f}")
    for category in res_by_category or {}:
        c = res_by_category[category]["correct"]
        t = res_by_category[category]["total"]
        cp = c / t * 100 if t else 0.0
        print(f"Category: {category} - Accuracy: {c}/{t} = {cp:.2f}")


"""Inference loop (shared across benchmarks)"""


def build_sample(bench_kind, dat, image_dir, task_instruction):
    """Return ``(image, prompt_text)`` for one record of the given benchmark."""
    if bench_kind == "hrbench":
        img_bytes = base64.b64decode(dat["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return image, dat["query"] + task_instruction
    if bench_kind == "blink":
        return dat["image"], dat["query"] + task_instruction
    if bench_kind == "vstar":
        img_path = os.path.join(image_dir, dat["image"])
        text = dat["text"].replace(
            "Answer with the option's letter from the given choices directly.",
            task_instruction,
        )
        return img_path, text
    raise ValueError(f"unknown bench_kind {bench_kind!r}")


def run_benchmark(
    model, processor, dataset, image_dir, bench_kind, steps, batch_size=1
):
    """Run inference over ``dataset`` in batches of ``batch_size`` and return a list of
    result dicts. Does NOT score or persist anything -- callers gather shards, score,
    and save once."""
    task_instruction = get_task_instruction(bench_kind)
    dataset = list(dataset)
    results = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for batch in tqdm(
        _chunked(dataset, batch_size),
        total=total_batches,
        desc=f"Evaluating {bench_kind}, decoding by steps={steps}",
    ):
        built = [
            build_sample(bench_kind, dat, image_dir, task_instruction) for dat in batch
        ]
        images = [b[0] for b in built]
        texts = [b[1] for b in built]
        outputs = run_inference(model, processor, images, texts, steps)
        for dat, out in zip(batch, outputs):
            results.append(
                {
                    "id": dat["question_id"],
                    "prediction": [out],
                    "label": dat["label"],
                    "category": dat.get("category", "overall"),
                }
            )
    return results


def _gpu_worker(
    gpu_id,
    shard,
    image_dir,
    bench_kind,
    model_path,
    steps,
    max_new_tokens,
    batch_size,
    image_pixels,
    queue,
):
    """Subprocess entry point: pin to one GPU, load a model replica, run one shard.

    ``CUDA_VISIBLE_DEVICES`` is set before any CUDA initialization so ``device_map=
    "auto"`` places the whole replica on this single card. Globals reset to module
    defaults under spawn, so the token-budget globals are re-applied here."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global MAX_NEW_TOKENS, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS
    MAX_NEW_TOKENS = max_new_tokens
    IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS = image_pixels
    model, processor = load_model_and_processor(model_path, steps)
    results = run_benchmark(
        model, processor, shard, image_dir, bench_kind, steps, batch_size
    )
    queue.put((gpu_id, results))


def run_data_parallel(
    dataset,
    image_dir,
    bench_kind,
    model_path,
    num_gpus,
    steps,
    max_new_tokens,
    batch_size,
):
    """Shard ``dataset`` round-robin across ``num_gpus`` worker processes, gather and
    re-interleave their predictions back into the original sample order."""
    dataset = list(dataset)
    n = max(1, min(num_gpus, len(dataset)))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for gpu_id in range(n):
        shard = dataset[gpu_id::n]
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                shard,
                image_dir,
                bench_kind,
                model_path,
                steps,
                max_new_tokens,
                batch_size,
                (IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS),
                queue,
            ),
        )
        p.start()
        procs.append(p)

    collected: dict[int, list] = {}
    for _ in range(n):
        gpu_id, shard_results = queue.get()
        collected[gpu_id] = shard_results
    for p in procs:
        p.join()

    return _interleave_shards([collected[i] for i in range(n)])


def _interleave_shards(shards):
    """Re-merge round-robin shards (``dataset[i::n]``) back into the original sample
    order: shard ``i`` holds samples ``i, i+n, i+2n, ...``."""
    results = []
    max_len = max((len(s) for s in shards), default=0)
    for j in range(max_len):
        for s in shards:
            if j < len(s):
                results.append(s[j])
    return results


def evaluate_benchmark(
    model, processor, dataset, image_dir, out_dir, bench_kind, args, model_path
):
    """Evaluate one benchmark: run (single- or multi-GPU), score, then save once."""
    print(f"Evaluating {bench_kind}")
    os.makedirs(out_dir, exist_ok=True)
    steps = args.steps
    out_file = os.path.join(out_dir, f"steps{steps:03d}.json")

    if args.num_gpus and args.num_gpus > 1:
        results = run_data_parallel(
            dataset,
            image_dir,
            bench_kind,
            model_path,
            args.num_gpus,
            steps,
            args.max_new_tokens,
            args.batch_size,
        )
    else:
        results = run_benchmark(
            model, processor, dataset, image_dir, bench_kind, steps, args.batch_size
        )

    correct, total, res_by_category = score_results(results)
    # Persist the full result set once, after the whole benchmark has finished.
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    _print_accuracy(steps, correct, total, res_by_category)


"""Data Loaders (read straight from the official HuggingFace repos)"""


def load_vstar_dataset(args, run_name):
    """V*Bench: 191 questions in ``test_questions.jsonl`` referencing images under the
    same repo (``direct_attributes/`` + ``relative_position/``). Official repo:
    ``craigwu/vstar_bench``."""
    local = resolve_repo(args.dataset)
    with open(os.path.join(local, "test_questions.jsonl"), "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    image_dir = local
    out_dir = os.path.join(args.output_dir, "VSTAR", run_name)
    return data, image_dir, out_dir, "vstar"


def load_hr_bench_dataset(args, run_name, split):
    """HR-Bench (VLMEvalKit format): one parquet per split with base64 ``image`` and
    ``A``-``D`` option columns. ``split`` is ``"4k"`` or ``"8k"``. Official repo:
    ``DreamMr/HR-Bench``."""
    parquet = resolve_file(args.dataset, f"hr_bench_{split}.parquet")
    df = pd.read_parquet(parquet)
    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "question_id": int(row["index"]),
                "image": row["image"],
                "query": row["question"]
                + "\nOptions:\n"
                + "(A)"
                + row["A"]
                + "\n"
                + "(B)"
                + row["B"]
                + "\n"
                + "(C)"
                + row["C"]
                + "\n"
                + "(D)"
                + row["D"],
                "label": row["answer"],
                "category": row.get("category", "overall"),
            }
        )
    out_dir = os.path.join(args.output_dir, f"HRBench{split}", run_name)
    return data, None, out_dir, "hrbench"


def load_blink_dataset(args, run_name):
    """BLINK: one HF config per task (val split), each with up to 4 images and a
    multiple-choice question. Official repo: ``BLINK-Benchmark/BLINK`` (14 task configs).
    From the hub ``get_dataset_config_names`` returns only real configs; from a local
    snapshot it can surface the non-config ``assets`` dir, so drop it explicitly."""
    configs = [c for c in get_dataset_config_names(args.dataset) if c != "assets"]
    processed_data = []
    for config in sorted(configs):
        # load_dataset returns a Union pyre can't index; the val split is a Dataset.
        ds: Any = load_dataset(args.dataset, config, split="val")
        for dat in ds:
            choices = dat["choices"]
            letters = string.ascii_uppercase
            option_string = ""
            for letter, choice in zip(letters, choices):
                option_string += f"{letter}. {choice}\n"
            if len(dat["answer"]) > 1:
                ans = dat["answer"][1].upper()
            else:
                ans = dat["answer"][0].upper()
            images = []
            for k in ["image_1", "image_2", "image_3", "image_4"]:
                if k in dat and dat[k] is not None:
                    images.append(dat[k])
            question = dat["question"] + "\nOptions:\n" + option_string
            processed_data.append(
                {
                    "question_id": dat["idx"],
                    "image": images,
                    "query": question,
                    "label": ans,
                    "category": config,
                }
            )
    out_dir = os.path.join(args.output_dir, "BLINK", run_name)
    return processed_data, None, out_dir, "blink"


def make_parser():
    """Common argument parser shared by every benchmark entry module. Benchmark-specific
    defaults (the HuggingFace repo id, and e.g. the HR-Bench pixel budget) are applied via
    ``parser.set_defaults(...)`` in the entry module before ``parse_args``."""
    p = argparse.ArgumentParser(description="VisReflect image-benchmark eval")
    p.add_argument(
        "--model_path",
        required=True,
        help="checkpoint dir or HuggingFace model id",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="benchmark source: HuggingFace dataset repo id (e.g. craigwu/vstar_bench) "
        "or a local dir. Default set per benchmark in the entry module.",
    )
    p.add_argument("--output_dir", default="./visreflect_eval_results")
    p.add_argument(
        "--steps",
        type=int,
        default=1,
        help="VTK reflection steps per generation (0 = vanilla Qwen2.5-VL baseline)",
    )
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument(
        "--image_min_pixels",
        type=int,
        default=None,
        help="min vision-token budget per image, in PIXELS (1 token = 28*28 px). "
        "None = qwen_vl_utils default (4*28*28). Per-benchmark default set in the "
        "entry module.",
    )
    p.add_argument(
        "--image_max_pixels",
        type=int,
        default=None,
        help="max vision-token budget per image, in PIXELS (1 token = 28*28 px). "
        "None = qwen_vl_utils default (16384*28*28). Per-benchmark default set in the "
        "entry module (HR-Bench bakes in 16384*28*28 / 4096*28*28).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="per-GPU inference batch size (>1 batches samples; high-res benchmarks "
        "may OOM)",
    )
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="single-host in-process data-parallel GPUs; >1 shards the benchmark "
        "across one spawned replica per GPU. Ignored under torchrun, where "
        "parallelism comes from the world size (one rank per GPU).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap samples (smoke test)",
    )
    return p


def _apply_runtime_globals(args):
    global MAX_NEW_TOKENS, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS
    MAX_NEW_TOKENS = args.max_new_tokens
    IMAGE_MIN_PIXELS = args.image_min_pixels
    IMAGE_MAX_PIXELS = args.image_max_pixels


def _run_distributed(args, bench_kind, loader):
    """torchrun path: one rank per GPU. Each rank pins ``LOCAL_RANK``, holds a full
    replica, evaluates its round-robin shard ``dataset[rank::world]``, then
    ``all_gather_object`` sends predictions to rank 0, which interleaves, scores, and
    saves. Parallelism comes from the world size; ``--num_gpus`` is ignored."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Long timeout: rank 0 alone stages model + datasets while other ranks wait at the
        # barrier; the default 10-min collective timeout could fire mid-stage.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=8))

    # Prefetch model + dataset on rank 0 only, then barrier so every rank reads the shared
    # HuggingFace cache instead of racing to download the same files. Single-node, so rank 0
    # is enough.
    run_name = derive_run_name(args.model_path)
    if rank == 0:
        resolve_model(args.model_path)
        loader(args, run_name)
    dist.barrier()

    model_path = resolve_model(args.model_path)  # cache hit on all ranks
    model, processor = load_model_and_processor(
        model_path, args.steps, device_map={"": local_rank}
    )

    dataset, image_dir, out_dir, _ = loader(args, run_name)
    dataset = list(dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    shard = dataset[rank::world]
    local_results = run_benchmark(
        model, processor, shard, image_dir, bench_kind, args.steps, args.batch_size
    )
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local_results)
    if rank == 0:
        results = _interleave_shards(gathered)
        correct, total, res_by_category = score_results(results)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"steps{args.steps:03d}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"=== {bench_kind} ===")
        _print_accuracy(args.steps, correct, total, res_by_category)
        print(f"Results written to {out_file}")
    dist.barrier()
    dist.destroy_process_group()


def _run_single_process(args, bench_kind, loader):
    """Single-host path: one process, optional in-process data parallelism via
    ``--num_gpus`` (``multiprocessing.spawn``, one replica per GPU)."""
    model_path = resolve_model(args.model_path)
    run_name = derive_run_name(args.model_path)

    # In data-parallel mode each worker loads its own replica; skip the main-process
    # load entirely (it would needlessly occupy a card).
    multi_gpu = args.num_gpus and args.num_gpus > 1
    model = processor = None
    if not multi_gpu:
        model, processor = load_model_and_processor(model_path, args.steps)

    dataset, image_dir, out_dir, _ = loader(args, run_name)
    if args.limit is not None:
        dataset = list(dataset)[: args.limit]
    evaluate_benchmark(
        model, processor, dataset, image_dir, out_dir, bench_kind, args, model_path
    )
    print(f"Results written under {out_dir}")


def run_eval(args, bench_kind, loader):
    """Evaluate ONE benchmark. ``loader`` is ``loader(args, run_name) -> (dataset,
    image_dir, out_dir, ds_name)``. Dispatches on ``WORLD_SIZE``: torchrun ->
    rank-aware data parallel; otherwise single process (optional ``--num_gpus`` spawn)."""
    _apply_runtime_globals(args)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_distributed(args, bench_kind, loader)
    else:
        _run_single_process(args, bench_kind, loader)
