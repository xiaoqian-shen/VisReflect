"""Shared machinery for the VisReflect VIDEO-benchmark evals (videomme, mlvu, mvbench).

Mirrors the image ``image_eval`` module and reuses it for model loading, VTK-marker
suppression, and MCQ scoring. Video specifics live here: self-staging (download the
benchmark repo from HuggingFace, unzip its video shards to local disk, auto-discover the
extracted video files by filename), per-benchmark record builders, and single-video
inference.

Benchmarks are read from their official HuggingFace repos (``--data_dir``, e.g.
``lmms-lab/Video-MME``); an existing local directory is also accepted. Extraction goes to
``$VISREFLECT_VIDEO_CACHE`` (default ``/tmp/visreflect_videobench``). Results are written to
the local ``--output_dir``.

``run_video_eval`` evaluates ONE benchmark and dispatches on ``WORLD_SIZE``: under torchrun
it runs rank-aware data parallel (one rank per GPU, each rank evaluates
``records[rank::world]``, gathered to rank 0); otherwise a plain single process. One video =
one sequence (no packing).
"""

import json
import logging
import os
import random
import string
import zipfile
from argparse import ArgumentParser
from datetime import timedelta

import torch
import torch.distributed as dist
from decord import VideoReader
from pyarrow import parquet as pq
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from evaluation.image_eval import (
    _interleave_shards,
    _print_accuracy,
    _vtk_suppress_ids,
    derive_run_name,
    load_model_and_processor,
    normalize_label,
    score_results,
)
from evaluation.hf_data import resolve_model, resolve_repo


logger: logging.Logger = logging.getLogger(__name__)

# Benchmark video shards are downloaded + extracted here once per host, guarded by a
# marker file so other ranks / reruns skip re-extraction.
LOCAL_STAGE_ROOT = os.environ.get("VISREFLECT_VIDEO_CACHE", "/tmp/visreflect_videobench")
# Denylist (not allowlist) so ANY video container format resolves -- MVBench mixes
# .mp4/.avi/.mkv/.webm/.mpeg/... across archives. Frame images and metadata are excluded
# so frame-dir tasks (tvqa/episodic_reasoning) and json/parquet don't pollute the index;
# anything else is treated as a candidate video and decode is attempted (failures are
# caught per-video in run_records).
NON_VIDEO_EXTS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif",
        ".gif",
        ".json",
        ".txt",
        ".md",
        ".csv",
        ".tsv",
        ".srt",
        ".vtt",
        ".ass",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".pkl",
        ".bin",
        ".parquet",
        ".zip",
        ".tar",
        ".gz",
        ".log",
        ".yaml",
        ".yml",
        ".html",
        ".lock",
    }
)


"""Staging"""


def stage_and_extract(source, bench_name):
    """Download the benchmark repo to local disk and unzip every ``*.zip`` into a stable
    per-benchmark extract dir. ``source`` is a HuggingFace dataset repo id or a local dir.
    Returns ``(downloaded_dir, extract_dir)``. Idempotent: a ``.extracted`` marker guards
    re-extraction."""
    downloaded = resolve_repo(source)
    extract_dir = os.path.join(LOCAL_STAGE_ROOT, bench_name)
    os.makedirs(extract_dir, exist_ok=True)
    marker = os.path.join(extract_dir, ".extracted")
    if not os.path.exists(marker):
        for root, _, files in os.walk(downloaded):
            for fname in files:
                if fname.endswith(".zip"):
                    logger.info("[stage] unzip %s -> %s", fname, extract_dir)
                    with zipfile.ZipFile(os.path.join(root, fname)) as zf:
                        zf.extractall(extract_dir)
        with open(marker, "w") as f:
            f.write("done\n")
    return downloaded, extract_dir


def build_video_index(extract_dir):
    """Index every extracted video file by basename AND stem, so a benchmark's ``video``
    field resolves whether it is ``x.mp4``, ``x``, or ``subdir/x.mp4``. First match wins
    on collision."""
    index: dict[str, str] = {}
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if ext.lower() in NON_VIDEO_EXTS or fname.startswith("."):
                continue
            full = os.path.join(root, fname)
            index.setdefault(fname, full)
            index.setdefault(stem, full)
    return index


def _lookup_video(index, video_field):
    base = os.path.basename(str(video_field))
    return index.get(base) or index.get(os.path.splitext(base)[0])


def _find_file(root, suffix):
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith(suffix):
                return os.path.join(dirpath, fname)
    return None


def _find_dir(root, name):
    for dirpath, dirs, _ in os.walk(root):
        if name in dirs:
            return os.path.join(dirpath, name)
    return None


def _letter_from_answer(answer, choices):
    """Coerce a gold answer into an option letter. Handles a bare letter (VideoMME) or
    the full option text (MLVU / MVBench)."""
    s = str(answer).strip()
    if len(s) == 1 and s.upper() in string.ascii_uppercase:
        return s.upper()
    if s in choices:
        return chr(ord("A") + choices.index(s))
    return normalize_label(s)


"""Per-benchmark record builders -> list of {id, video, question, choices, label, category}"""


def build_videomme_records(args):
    downloaded, extract_dir = stage_and_extract(args.data_dir, "videomme")
    index = build_video_index(extract_dir)
    parquet = _find_file(downloaded, "test-00000-of-00001.parquet") or _find_file(
        downloaded, ".parquet"
    )
    df = pq.read_table(parquet).to_pandas()
    records, missing = [], 0
    for r in df.itertuples():
        ytid = (
            str(r.url).split("watch?v=")[-1]
            if isinstance(r.url, str)
            else str(r.videoID)
        )
        vpath = _lookup_video(index, ytid) or _lookup_video(index, str(r.videoID))
        if not vpath:
            missing += 1
            continue
        choices = list(r.options)
        records.append(
            {
                "id": f"{r.video_id}_{r.question_id}",
                "video": vpath,
                "question": r.question,
                "choices": choices,
                "label": _letter_from_answer(r.answer, choices),
                "category": str(getattr(r, "duration", "overall")),
            }
        )
    if missing:
        logger.warning("[videomme] %d questions skipped (video not found)", missing)
    return records


def build_mlvu_records(args):
    downloaded, extract_dir = stage_and_extract(args.data_dir, "mlvu")
    index = build_video_index(extract_dir)
    parquet = _find_file(downloaded, ".parquet")
    df = pq.read_table(parquet).to_pandas()
    records, missing = [], 0
    for i, r in enumerate(df.itertuples()):
        vpath = _lookup_video(index, r.video_name)
        if not vpath:
            missing += 1
            continue
        choices = list(r.candidates)
        category = str(getattr(r, "task_type", None) or "overall")
        records.append(
            {
                "id": str(getattr(r, "question_id", None) or f"{r.video_name}_{i}"),
                "video": vpath,
                "question": r.question,
                "choices": choices,
                "label": _letter_from_answer(r.answer, choices),
                "category": category,
            }
        )
    if missing:
        logger.warning("[mlvu] %d questions skipped (video not found)", missing)
    return records


def build_mvbench_records(args):
    downloaded, extract_dir = stage_and_extract(args.data_dir, "mvbench")
    index = build_video_index(extract_dir)
    json_dir = _find_dir(downloaded, "json")
    records, missing = [], 0
    for jf in sorted(os.listdir(json_dir)):
        if not jf.endswith(".json"):
            continue
        task = os.path.splitext(jf)[0]
        with open(os.path.join(json_dir, jf)) as f:
            items = json.load(f)
        for i, item in enumerate(items):
            vpath = _lookup_video(index, item["video"])
            if not vpath:
                # frame-dir tasks (e.g. episodic_reasoning/tvqa) have no single video file;
                # whole-video filename-indexed mode skips them by design.
                missing += 1
                continue
            choices = item["candidates"]
            records.append(
                {
                    "id": f"{task}_{i}",
                    "video": vpath,
                    "question": item["question"],
                    "choices": choices,
                    "label": _letter_from_answer(item["answer"], choices),
                    "category": task,
                }
            )
    if missing:
        logger.warning("[mvbench] %d questions skipped (no single video file)", missing)
    return records


"""Inference"""


def _build_input_text(question, choices):
    candidates = "".join(f"{chr(ord('A') + i)}. {c}\n" for i, c in enumerate(choices))
    return (
        f"{question}\n{candidates}"
        "Answer with the option's letter from the given choices directly."
    )


# The exact answer prefix the model is trained to emit after the reflection block -- keep in
# sync with parse_grounded_video_record's `clean_answer = f"The answer is: {X}"`.
_ANSWER_PREFIX = "The answer is:"


def _answer_constraint_fn(tokenizer, config, prompt_len, num_options):
    """Build a ``prefix_allowed_tokens_fn`` that makes the model emit ONLY a valid option
    letter after the reflection block.

    The decode loop still force-generates ``<BOR><VR>..<EOR>`` on its own; once ``<EOR>`` is
    out, this teacher-forces the training answer prefix (``The answer is:``) token by token,
    then at the letter slot restricts the vocabulary to the option-letter tokens (so greedy =
    argmax over A.., exactly matching training), then forces ``<|im_end|>``. Nothing outside
    this template can be produced, so stray characters / repetition degeneration are impossible.
    """
    bor = config.vtk_start_id
    eor = config.vtk_end_id
    vr = config.vtk_id
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end, int) or im_end < 0:
        im_end = tokenizer.eos_token_id

    prefix_ids = tokenizer(_ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    letter_ids = []
    for i in range(min(num_options, 26)):
        letter = chr(ord("A") + i)
        # The letter token AS IT FOLLOWS the prefix (handles the leading space in ": A").
        full = tokenizer(f"{_ANSWER_PREFIX} {letter}", add_special_tokens=False)[
            "input_ids"
        ]
        if full[: len(prefix_ids)] == prefix_ids and len(full) > len(prefix_ids):
            letter_ids.append(full[len(prefix_ids)])
        else:
            letter_ids.append(
                tokenizer(f" {letter}", add_special_tokens=False)["input_ids"][0]
            )

    def fn(batch_id, input_ids):
        gen = input_ids[prompt_len:].tolist()
        # Answer always begins with <BOR>; force it so the reflection window starts.
        if not gen:
            return [bor]
        # Inside the reflection window: the decode loop force-overrides these to <VR>/<EOR>.
        if eor not in gen:
            return [vr]
        after = gen[len(gen) - 1 - gen[::-1].index(eor) + 1 :]
        k = len(after)
        if k < len(prefix_ids):
            return [prefix_ids[k]]  # teacher-force "The answer is:"
        if k == len(prefix_ids):
            return letter_ids  # letter slot: argmax over the option letters only
        return [im_end]  # letter emitted -> stop

    return fn


def _sample_nframes(video_path, fps, max_frames):
    """Number of frames to sample: ~``fps`` frames/sec, capped at ``max_frames``
    (uniformly subsampled only when 1fps would exceed it), and never more than the clip
    length. Rounded down to an even count (Qwen temporal_patch_size=2). We do NOT raise
    fps to reach ``max_frames`` on short clips; over-requesting frames makes
    process_vision_info / decord error -> blank output."""
    try:
        vr = VideoReader(video_path, num_threads=1)
        total = len(vr)
        video_fps = vr.get_avg_fps() or fps
    except Exception:
        return max_frames
    n = round(total / max(video_fps, 1e-6) * fps)
    n = min(n, max_frames, total)
    if n >= 2:
        n = (n // 2) * 2
    return max(1, n)


def run_video_inference(model, processor, record, args):
    """Single-video inference -> decoded string. One video per sequence."""
    patch = model.config.vision_config.patch_size
    nframes = _sample_nframes(record["video"], args.fps, args.max_frames)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": record["video"],
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "total_pixels": args.total_pixels,
                    "nframes": nframes,
                },
                {
                    "type": "text",
                    "text": _build_input_text(record["question"], record["choices"]),
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=patch)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    ).to(model.device)

    # Answer decoding for the VTK path. Default (constrain_answer): after the forced
    # <BOR><VR>..<EOR> block, teacher-force the training prefix (_ANSWER_PREFIX, empty by
    # default) and restrict the next token to the option letters -> the model can ONLY emit a
    # valid letter (argmax over A..), exactly matching training, so stray characters are
    # impossible. If disabled,
    # fall back to free generation with the repetition knobs (breaks loops but doesn't hard-ban
    # the garbage vocabulary; repetition_penalty also mildly penalizes prompt tokens incl. the
    # option letters, so keep it modest).
    anti_degen = {}
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        anti_degen["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        anti_degen["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    with torch.no_grad():
        if args.steps >= 1 and args.constrain_answer:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                steps=[args.steps],
                prefix_allowed_tokens_fn=_answer_constraint_fn(
                    processor.tokenizer,
                    model.config,
                    prompt_len=inputs.input_ids.shape[1],
                    num_options=len(record["choices"]),
                ),
            )
        elif args.steps >= 1:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                steps=[args.steps],
                **anti_degen,
            )
        else:
            gen_kwargs = {"max_new_tokens": args.max_new_tokens, **anti_degen}
            suppress = _vtk_suppress_ids(model, processor)
            if suppress:
                gen_kwargs["suppress_tokens"] = suppress
            generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]


def run_records(model, processor, records, args, desc):
    """Run inference over records; a decode failure on one video records "" (scored
    wrong) rather than aborting the whole shard."""
    results = []
    for rec in tqdm(records, desc=desc):
        try:
            out = run_video_inference(model, processor, rec, args)
        except Exception as e:
            logger.warning("inference failed for %s: %s", rec.get("video"), e)
            out = ""
        results.append(
            {
                "id": rec["id"],
                "prediction": [out],
                "label": rec["label"],
                "category": rec["category"],
            }
        )
    return results


"""Dispatch"""


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _maybe_sample(records, args):
    """Apply --limit: with --sample_seed set, randomly (seeded, reproducible) pick that many
    records for a representative spot-check; otherwise take the first --limit."""
    if args.limit is None:
        return records
    if args.sample_seed is not None:
        records = list(records)
        random.Random(args.sample_seed).shuffle(records)
    return records[: args.limit]


def make_video_parser():
    p = ArgumentParser(description="VisReflect video-benchmark eval")
    p.add_argument(
        "--model_path", required=True, help="checkpoint dir or HuggingFace model id"
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="benchmark source: HuggingFace dataset repo id (e.g. lmms-lab/Video-MME) "
        "or a local dir; default set per benchmark in the entry module",
    )
    p.add_argument("--output_dir", default="./visreflect_video_eval_results")
    p.add_argument(
        "--steps",
        type=int,
        default=1,
        help="VTK reflection steps per generation (0 = vanilla baseline)",
    )
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.3,
        help="penalize already-generated tokens to break repetitive degeneration "
        "(1.0 = off). Mildly penalizes prompt tokens too, so keep modest.",
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="forbid repeating any n-gram of this size (0 = off); kills '1 1 1 1'-style loops. "
        "Only used when --constrain_answer is False.",
    )
    p.add_argument(
        "--constrain_answer",
        type=_str2bool,
        default=True,
        help="after the <EOR> reflection block, teacher-force the answer prefix (empty by "
        "default) and restrict the next token to the option letters (argmax over A..) -- "
        "guarantees a valid letter and makes stray characters impossible. Set False to "
        "free-generate with the repetition knobs.",
    )
    p.add_argument("--fps", type=float, default=1.0, help="sample frames at this fps")
    p.add_argument(
        "--max_frames",
        type=int,
        default=256,
        help="cap on sampled frames; if fps-sampling yields more, uniformly subsample to "
        "this (short clips keep their fps frame count -- fps is NOT raised to fill it)",
    )
    p.add_argument(
        "--total_pixels",
        type=int,
        default=16384 * 28 * 28,
        help="per-video total pixel budget across all frames (1 token = 28*28 px)",
    )
    p.add_argument(
        "--min_pixels",
        type=int,
        default=128 * 28 * 28,
        help="per-frame pixel floor (1 token = 28*28 px)",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=460800,
        help="per-frame pixel cap (1 token = 28*28 px)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap samples (smoke test)")
    p.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="if set, RANDOMLY sample --limit records (seeded, reproducible) instead of the "
        "first --limit -- for a representative format/quality spot-check",
    )
    return p


def _save_and_score(args, bench_name, run_name, results):
    out_dir = os.path.join(args.output_dir, bench_name.upper(), run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"steps{args.steps:03d}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    correct, total, res_by_category = score_results(results)
    # Machine-readable summary alongside the per-sample rows so downstream tooling reads the
    # score directly instead of re-deriving it.
    metrics = {
        "benchmark": bench_name,
        "steps": args.steps,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "by_category": {
            c: {"correct": v["correct"], "total": v["total"]}
            for c, v in res_by_category.items()
        },
    }
    with open(os.path.join(out_dir, f"steps{args.steps:03d}.metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"=== {bench_name} ===")
    _print_accuracy(args.steps, correct, total, res_by_category)
    print(f"Results written under {out_dir}")


def _run_distributed(args, bench_name, build_records):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Long timeout: rank 0 alone downloads + unzips the benchmark (MLVU ~280GB) while
        # the other ranks wait at the barrier below; the default 10-min collective timeout
        # would fire mid-stage and kill them.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=8))

    # Stage model + benchmark on rank 0 only (download + unzip), then barrier so every rank
    # reads the shared cache instead of racing to download / unzip the same files.
    run_name = derive_run_name(args.model_path)
    if rank == 0:
        resolve_model(args.model_path)
        build_records(args)
    dist.barrier()

    model_path = resolve_model(args.model_path)
    model, processor = load_model_and_processor(
        model_path, args.steps, device_map={"": local_rank}
    )
    records = build_records(args)
    records = _maybe_sample(records, args)
    shard = records[rank::world]
    local_results = run_records(
        model, processor, shard, args, desc=f"{bench_name} rank{rank}"
    )
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local_results)
    if rank == 0:
        results = _interleave_shards(gathered)
        _save_and_score(args, bench_name, run_name, results)
    dist.barrier()
    dist.destroy_process_group()


def _run_single_process(args, bench_name, build_records):
    model_path = resolve_model(args.model_path)
    run_name = derive_run_name(args.model_path)
    model, processor = load_model_and_processor(model_path, args.steps)
    records = build_records(args)
    records = _maybe_sample(records, args)
    results = run_records(model, processor, records, args, desc=bench_name)
    _save_and_score(args, bench_name, run_name, results)


def run_video_eval(args, bench_name, build_records):
    """Evaluate ONE video benchmark. ``build_records(args) -> list[record]`` self-stages
    its data. Dispatches on ``WORLD_SIZE`` (torchrun vs single process)."""
    if args.data_dir is None:
        raise ValueError(
            "--data_dir is required (set per-benchmark default in entry module)"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_distributed(args, bench_name, build_records)
    else:
        _run_single_process(args, bench_name, build_records)
