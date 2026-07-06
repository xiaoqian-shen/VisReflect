# VisReflect

<div align="center">
    <h4>
        <div>
          VisReflect: Latent Visual Reflection for Fine-Grained Perception in Long Visual Context
<div></div>

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat-square)](https://xiaoqian-shen.github.io/VisReflect/)
[![Paper](https://img.shields.io/badge/arXiv-2606.30288-b31b1b?style=flat-square)](https://arxiv.org/abs/2606.30288)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=flat-square)](https://huggingface.co/collections/shenxq/visreflect)

[**Xiaoqian Shen**](https://xiaoqian-shen.github.io/), [**Mohamed Elhoseiny**](https://cemse.kaust.edu.sa/profiles/mohamed-elhoseiny)

King Abdullah University of Science and Technology (KAUST)

</div>
    </h4>
</div>

## :rocket: Get Started

### ⚙️ Environment setup

```bash
git clone <this-repo> VisReflect && cd VisReflect
conda create -n visreflect python=3.12 -y && conda activate visreflect
pip install -r requirements.txt          # PyTorch 2.6.0 + CUDA 12.4
pip install flash-attn==2.7.4.post1 --no-build-isolation
export PYTHONPATH="$PWD:$PYTHONPATH"      # run modules from the repo root
```

## 🤗 Model Zoo

| Model | Base | Modality | Weights |
| :--- | :---: | :---: | :---: |
| **VisReflect-7B-Image** | Qwen2.5-VL-7B | high-resolution image | [🤗 shenxq/VisReflect-7B-Image](https://huggingface.co/shenxq/VisReflect-7B-Image) |
| **VisReflect-7B-Video** | Qwen2.5-VL-7B | long video | [🤗 shenxq/VisReflect-7B-Video](https://huggingface.co/shenxq/VisReflect-7B-Video) |

`--model_path` / `MODEL_ID` accept a HuggingFace id (downloaded automatically) or a local
checkpoint dir.

### 📊 Evaluation

Benchmarks are read straight from their official HuggingFace repos (or a local dir), and
results are written to `--output_dir`. Pick a benchmark with `--task`:

| Modality | `--task` | Benchmark (HF dataset) |
| :--- | :--- | :--- |
| Image | `vstar` · `hrbench4k` · `hrbench8k` · `blink` | [V\*](https://huggingface.co/datasets/craigwu/vstar_bench) · [HR-Bench-4K](https://huggingface.co/datasets/DreamMr/HR-Bench) · [HR-Bench-8K](https://huggingface.co/datasets/DreamMr/HR-Bench) · [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) |
| Video | `videomme` · `mlvu` · `mvbench` | [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) · [MLVU](https://huggingface.co/datasets/MLVU/MVLU) · [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) |

```bash
# single benchmark (steps>=1 = VTK reflection; steps=0 = vanilla baseline)
python -m evaluation.eval_config --task vstar \
    --model_path shenxq/VisReflect-7B-Image --steps 1 --output_dir ./results

# multi-GPU (one rank/GPU; required for the video benchmarks)
torchrun --standalone --nproc_per_node=8 -m evaluation.eval_config --task videomme \
    --model_path shenxq/VisReflect-7B-Video --steps 4 --output_dir ./results

# or loop several tasks via the launchers
BENCHMARKS=vstar,hrbench8k,blink MODEL_PATH=shenxq/VisReflect-7B-Image ./scripts/run_image_eval.sh
BENCHMARKS=videomme,mlvu,mvbench MODEL_PATH=shenxq/VisReflect-7B-Video ./scripts/run_video_eval.sh
```

### 🔥 Training

#### 🖼️ Image SFT

Data: [deepcs233/Visual-CoT](https://huggingface.co/datasets/deepcs233/Visual-CoT)
(`viscot_363k.json`).

```bash
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct DATA_PATH=/path/viscot_363k.json \
    IMAGE_FOLDER=/path/cot_images ./scripts/run_image_train.sh
```

#### 🎬 Video SFT

Data (convert to the record format below): [G-VideoQA](https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/blob/main/G-VideoQA-gpt4o-mini-anno.json),
[PLM-Video-Auto](https://huggingface.co/datasets/facebook/PLM-Video-Auto),
[NExT-GQA](https://huggingface.co/datasets/jinyoungkim/NExT-GQA). Each record is a local JSON
entry with a video, a QA pair, and the temporal span of the clue region:

```json
{"video": "sub/dir/clip.mp4", "question": "...", "answer": "...", "temporal_span": [3.0, 7.5]}
```

`num_clue_frames` frames sampled from `temporal_span` form the `<VR>` alignment target; the
full clip is the question context (one video = one sequence). Init from the image checkpoint
for continual image→video training.

```bash
MODEL_ID=shenxq/VisReflect-7B-Image DATA_PATH=/path/video_records.json \
    VIDEO_FOLDER=/path/videos ./scripts/run_video_train.sh
```

## ✏️ Citation

```bibtex
@article{shen2026visreflect,
  title={VisReflect: Latent Visual Reflection for Fine-Grained Perception in Long Visual Context},
  author={Shen, Xiaoqian and Elhoseiny, Mohamed},
  journal={European Conference on Computer Vision},
  year={2026}
}
```
