import re

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
    VTK_END_TOKEN,
    VTK_START_TOKEN,
    VTK_TOKEN,
)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r"\n?" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r"\n?" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)


def replace_vtk_tokens(input_string, vtk_counts):
    """Prepend the visual-reflection block(s) to an assistant turn, just before the
    answer: for each bbox crop, VTK_START + N*VTK_TOKEN + VTK_END (i.e. <BOR><VR>..<EOR>)
    in bbox order, where N is the crop's merged-patch count. A count of 0 (bbox area >=
    threshold) contributes no block. Returns (new_text, consumed_counts)."""
    input_string = input_string.replace("<answer> ", "The answer is: ").replace(
        " </answer>", ""
    )
    prefix = ""
    consumed = []
    for count in vtk_counts:
        consumed.append(count)
        if count > 0:
            prefix += VTK_START_TOKEN + VTK_TOKEN * count + VTK_END_TOKEN
    if prefix:
        return prefix + input_string, consumed
    if consumed:
        # bbox(es) present but all skipped (area >= threshold)
        return input_string.strip(), consumed
    return input_string, consumed


def llava_to_openai(conversations, is_video=False, vtk_counts_list=None):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    all_vtk_counts = []
    vtk_offset = 0
    for conversation in conversations:
        role = role_mapping.get(conversation["from"], conversation["from"])
        content = replace_image_tokens(conversation["value"], is_video=is_video)
        # Reflection blocks are inserted only on assistant turns. Visual-CoT records
        # have a single assistant turn that owns all the record's bboxes, so it
        # consumes the remaining counts in order.
        if role == "assistant":
            remaining = (
                vtk_counts_list[vtk_offset:] if vtk_counts_list is not None else []
            )
            content, consumed = replace_vtk_tokens(content, remaining)
            vtk_offset += len(consumed)
            all_vtk_counts.extend(consumed)
        transformed_data.append({"role": role, "content": content})
    return transformed_data, all_vtk_counts


def resolve_dataset_specs(loaded, default_image_folder):
    """Accept either a direct records list (each item has 'conversations') or a
    meta-manifest (list of {data_path, image_folder, ds_name}); return a list of
    (data, image_folder, ds_name) tuples, where `data` is a records list (direct file)
    or a path string (meta entry). This lets --data_path point straight at one merged
    records file, with no separate meta-manifest."""
    if loaded and isinstance(loaded[0], dict) and "conversations" in loaded[0]:
        return [(loaded, default_image_folder, "dataset")]
    return [(m["data_path"], m["image_folder"], m["ds_name"]) for m in loaded]


# --------------------------------------------------------------------------- #
# Raw Visual-CoT (deepcs233/Visual-CoT `viscot_363k.json`) -> internal record
#
# A raw record is {"image": ["cot/<ds>/<f>.jpg", "cot/<ds>/<f>.jpg###[x1,y1,x2,y2]"],
# "conversations": [human(question + "Please provide the bounding box ..."),
# gpt(normalized [0,1] bbox), human("<image>"), gpt(answer)]}. The region is read from the
# model-target gpt turn (already normalized to the original image, so no image is opened);
# the answer is wrapped `<answer> ... </answer>` -- `replace_vtk_tokens` above turns the
# assistant turn into `<BOR><VR>..<EOR>The answer is: <answer>`. This lets `--data_path`
# point straight at the raw viscot_363k.json; the dataloader converts on load.
# --------------------------------------------------------------------------- #
_VISCOT_BBOX_INSTRUCTION = re.compile(
    r"\s*Please provide the bounding box coordinate.*", re.IGNORECASE | re.DOTALL
)
_VISCOT_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
_VISCOT_PURE_BBOX = re.compile(
    r"^[\[\(]?\s*(?:[-+]?\d+(?:\.\d+)?\s*,\s*){3}[-+]?\d+(?:\.\d+)?\s*[\]\)]?$"
)


def _is_raw_viscot(rec) -> bool:
    """Raw Visual-CoT records carry the region inside the conversation and have NO
    top-level `bboxes`; internal VisReflect records always do."""
    return isinstance(rec, dict) and "conversations" in rec and "bboxes" not in rec


def viscot_raw_to_record(rec: dict):
    """One raw viscot_363k.json record -> internal {image, bboxes, conversations}. Returns
    None to skip a record with no parseable answer (a handful are pure temporal grounding)."""
    image = rec["image"]
    base_image = image[0] if isinstance(image, list) else image
    convs = rec.get("conversations", [])

    bboxes = []
    for c in convs:
        if c.get("from") == "gpt" and _VISCOT_PURE_BBOX.match(c.get("value", "").strip()):
            nums = [float(x) for x in _VISCOT_NUM.findall(c["value"])[:4]]
            if len(nums) == 4:
                bboxes.append([min(max(v, 0.0), 1.0) for v in nums])

    question = None
    for c in convs:
        if c.get("from") == "human":
            q = _VISCOT_BBOX_INSTRUCTION.sub("", c.get("value", "")).strip()
            question = q if "<image>" in q else "<image>\n" + q
            break

    answer = None
    for c in reversed(convs):
        if c.get("from") == "gpt":
            v = c.get("value", "").strip()
            if v and not _VISCOT_PURE_BBOX.match(v):
                answer = v
                break

    if question is None or answer is None:
        return None
    return {
        "image": base_image,
        "bboxes": bboxes,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": f"\n<answer> {answer} </answer>"},
        ],
    }


def normalize_records(records: list) -> list:
    """Accept either internal VisReflect records (top-level `bboxes`) or raw Visual-CoT
    (viscot_363k.json) records; return internal records. Detection is by the first record,
    so a file is all-raw or all-internal. Raw records with no answer are dropped."""
    if not records or not _is_raw_viscot(records[0]):
        return records
    converted = [viscot_raw_to_record(r) for r in records]
    return [r for r in converted if r is not None]


def get_vtk_crop_inputs(
    processor, pil_image, bboxes, min_pixels, max_pixels, area_threshold=1.0
):
    """Crop each (normalized xyxy) bbox from the full-resolution image, re-encode
    the crop through the Qwen image processor, and return:
      - vtk_pixel_values: [sum_patches, feat_dim] float, crops concatenated
      - vtk_grid_thw:     [num_kept_crops, 3] long
      - counts:           list[int] aligned with `bboxes`; merged-token count per
                          crop, 0 for bboxes skipped (area >= area_threshold).
    Cropping the original image (instead of slicing patches from the downsized
    full-image encoding) lets the ViT see the region at higher fidelity.
    """
    image_processor = processor.image_processor
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size
    factor = patch_size * merge_size
    feat_dim = 3 * image_processor.temporal_patch_size * patch_size * patch_size
    w, h = pil_image.size

    pixel_values_list = []
    grid_list = []
    counts = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        # bboxes are normalized xyxy, but a few records carry out-of-range (e.g. dude
        # coords up to ~9.8) or inverted corners; clamp to [0,1] and order the corners
        # so pil_image.crop() always gets left<=right, top<=bottom.
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
        area = (x1 - x0) * (y1 - y0)
        # skip degenerate (corrupt -> zero area after clamp) and too-large regions
        if area <= 0 or area >= area_threshold:
            counts.append(0)
            continue

        bx0 = min(int(round(x0 * w)), w - 1)
        by0 = min(int(round(y0 * h)), h - 1)
        bx1 = min(w, max(bx0 + 1, int(round(x1 * w))))
        by1 = min(h, max(by0 + 1, int(round(y1 * h))))

        crop = pil_image.crop((bx0, by0, bx1, by1))
        cw, ch = crop.size
        # smart_resize raises on aspect ratios >= 200 (degenerate sliver crops from a
        # ~1px-thin bbox); skip those like other invalid crops -> no reflection block,
        # CE still computed. Legitimate thin crops (text lines, ratio < 200) pass.
        if min(cw, ch) <= 0 or max(cw, ch) / min(cw, ch) >= 200:
            counts.append(0)
            continue
        rh, rw = smart_resize(
            ch, cw, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        crop = crop.resize((rw, rh), Image.BICUBIC)
        out = image_processor(images=crop, do_resize=False, return_tensors="pt")
        grid = out["image_grid_thw"]
        pixel_values_list.append(out["pixel_values"])
        grid_list.append(grid)
        counts.append(int(grid.prod().item()) // (merge_size**2))

    if pixel_values_list:
        vtk_pixel_values = torch.cat(pixel_values_list, dim=0)
        vtk_grid_thw = torch.cat(grid_list, dim=0)
    else:
        vtk_pixel_values = torch.zeros((0, feat_dim), dtype=torch.float32)
        vtk_grid_thw = torch.zeros((0, 3), dtype=torch.long)
    return vtk_pixel_values, vtk_grid_thw, counts


def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(
    image_path, min_pixel, max_pixel, width, height, image_patch_size=16
):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]

    image_input, _ = process_vision_info(messages, image_patch_size=image_patch_size)

    return image_input[0]


# --------------------------------------------------------------------------- #
# Video helpers (visual-reflection alignment on video)
# --------------------------------------------------------------------------- #
SPATIAL_MERGE_SIZE = 2
VIDEO_MAX_TOKEN_NUM = 768
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4


def sample_clue_frame_indices(
    intervals, video_fps: float, total_frames: int, num_frames: int = 3
) -> list[int]:
    """Uniformly sample up to ``num_frames`` frame indices spanning the clue interval(s).

    A single interval is a ``linspace`` over its ``[lo, hi]`` frame range; multiple intervals
    get frames apportioned by duration. Indices are clamped to ``[0, total_frames - 1]`` and
    de-duplicated, so the result has between 1 and ``num_frames`` indices.
    """
    ranges = []
    for lo, hi in intervals:
        f_lo = int(round(min(lo, hi) * video_fps))
        f_hi = int(round(max(lo, hi) * video_fps))
        ranges.append(
            (max(0, min(f_lo, total_frames - 1)), max(0, min(f_hi, total_frames - 1)))
        )
    total_span = sum(hi - lo for lo, hi in ranges) or 1
    candidates: list[int] = []
    for lo, hi in ranges:
        n = max(1, int(round(num_frames * (hi - lo) / total_span)))
        candidates.extend(int(round(x)) for x in np.linspace(lo, hi, n))
    candidates = sorted({max(0, min(i, total_frames - 1)) for i in candidates})
    if len(candidates) > num_frames:
        pick = np.linspace(0, len(candidates) - 1, num_frames).round().astype(int)
        candidates = [candidates[i] for i in pick]
    return candidates or [0]


def build_video_reflection_answer(answer: str, num_vtk_tokens: int) -> str:
    """Prepend ONE visual-reflection block (``<BOR>`` + N``<VR>`` + ``<EOR>``) to the answer,
    N being the merged-token count of the sampled clue frames. The video analogue of the
    per-bbox blocks ``replace_vtk_tokens`` inserts for images."""
    if num_vtk_tokens > 0:
        return VTK_START_TOKEN + VTK_TOKEN * num_vtk_tokens + VTK_END_TOKEN + answer
    return answer


def fetch_video(
    vr, min_pixels, max_pixels, image_patch_size, max_seq_len, fps=1.0, max_frames=128
):
    """Decode the full video (the question context) as a TCHW float tensor + metadata.

    Frames are sampled at ~``fps`` (clamped to ``[FPS_MIN_FRAMES, max_frames]`` and rounded to a
    multiple of ``FRAME_FACTOR``), then resized to a per-frame pixel budget derived from the
    sequence-length cap."""
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    nframes = total_frames / max(video_fps, 1e-6) * fps
    nframes = min(max(nframes, FPS_MIN_FRAMES), max_frames, total_frames)
    nframes = max(FRAME_FACTOR, int(round(nframes / FRAME_FACTOR)) * FRAME_FACTOR)
    nframes = min(nframes, total_frames)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )

    nframes, _, height, width = video.shape
    total_pixels = max_seq_len * image_factor * image_factor * 0.9
    frame_max_pixels = max(
        min(video_frame_max_pixels, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    frame_max_pixels = min(max_pixels, frame_max_pixels)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=frame_max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    return video, video_metadata
