"""Single entry point for every VisReflect benchmark eval.

Each benchmark is one row in the ``IMAGE_TASKS`` / ``VIDEO_TASKS`` registry below (its
official HuggingFace repo id + baked-in vision-token / frame budget). Pick one with
``--task``; the rest of the CLI is the shared image or video parser. Run with, e.g.::

    python -m evaluation.eval_config --task hrbench8k --model_path <checkpoint> --output_dir ./results
    torchrun --standalone --nproc_per_node=8 -m evaluation.eval_config --task videomme --model_path <ckpt>

``--steps N`` (N>=1) runs the VTK reflection decode; ``--steps 0`` is the vanilla
Qwen2.5-VL baseline (reflection markers suppressed).
"""

import argparse

from evaluation.image_eval import (
    load_blink_dataset,
    load_hr_bench_dataset,
    load_vstar_dataset,
    make_parser,
    run_eval,
)
from evaluation.video_eval import (
    build_mlvu_records,
    build_mvbench_records,
    build_videomme_records,
    make_video_parser,
    run_video_eval,
)


# HR-Bench bakes its vision-token budget into the defaults (16384 / 4096 tokens; raising it
# further did not help). vstar / blink use the qwen_vl_utils defaults.
_HRBENCH_PIXELS = {
    "image_max_pixels": 16384 * 28 * 28,
    "image_min_pixels": 4096 * 28 * 28,
}

# task -> (bench_kind for build_sample, HF dataset repo id, loader, arg defaults).
IMAGE_TASKS = {
    "vstar": ("vstar", "craigwu/vstar_bench", load_vstar_dataset, {}),
    "blink": ("blink", "BLINK-Benchmark/BLINK", load_blink_dataset, {}),
    "hrbench4k": (
        "hrbench",
        "DreamMr/HR-Bench",
        lambda a, run_name: load_hr_bench_dataset(a, run_name, "4k"),
        _HRBENCH_PIXELS,
    ),
    "hrbench8k": (
        "hrbench",
        "DreamMr/HR-Bench",
        lambda a, run_name: load_hr_bench_dataset(a, run_name, "8k"),
        _HRBENCH_PIXELS,
    ),
}

# Per-benchmark frame/pixel budget. 1 merged token = 28*28 px; total = per-video budget
# across frames, max/min = per-frame cap/floor.
_LONG_VIDEO_PIXELS = {
    "fps": 2,
    "max_frames": 512,
    "total_pixels": 20000 * 28 * 28,
    "max_pixels": 460800,
    "min_pixels": 64 * 28 * 28,
}

# task -> (HF dataset repo id, record builder, arg defaults).
VIDEO_TASKS = {
    "videomme": ("lmms-lab/Video-MME", build_videomme_records, _LONG_VIDEO_PIXELS),
    "mlvu": ("MLVU/MVLU", build_mlvu_records, _LONG_VIDEO_PIXELS),
    "mvbench": (
        "OpenGVLab/MVBench",
        build_mvbench_records,
        {
            "fps": 2,
            "max_frames": 256,
            "total_pixels": 16384 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
            "min_pixels": 16 * 28 * 28,
        },
    ),
}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--task",
        required=True,
        choices=list(IMAGE_TASKS) + list(VIDEO_TASKS),
    )
    ns, rest = pre.parse_known_args()

    if ns.task in IMAGE_TASKS:
        bench_kind, dataset, loader, defaults = IMAGE_TASKS[ns.task]
        parser = make_parser()
        parser.set_defaults(dataset=dataset, **defaults)
        args = parser.parse_args(rest)
        run_eval(args, bench_kind, loader)
    else:
        data_dir, builder, defaults = VIDEO_TASKS[ns.task]
        parser = make_video_parser()
        parser.set_defaults(data_dir=data_dir, **defaults)
        args = parser.parse_args(rest)
        run_video_eval(args, ns.task, builder)


if __name__ == "__main__":
    main()
