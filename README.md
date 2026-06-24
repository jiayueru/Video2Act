<div align="center">

# Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling

<a href="https://arxiv.org/abs/2512.03044"><img src="https://img.shields.io/badge/arXiv-2512.03044-b31b1b.svg"></a> &nbsp;
<a href="https://video2act.github.io/"><img src="https://img.shields.io/badge/Project-Video2Act-blue"></a> &nbsp;
<a href="https://github.com/robotwin-Platform/RoboTwin"><img src="https://img.shields.io/badge/Built%20on-RoboTwin-green"></a>

[Yueru Jia](https://jiayueru.github.io/)<sup>1,2*</sup>, [Jiaming Liu](https://liujiaming1996.github.io/)<sup>1,2*</sup>, [Shengbang Liu](https://liushb9.github.io/)<sup>3*</sup>, [Rui Zhou](https://zhourui9813.github.io/)<sup>4</sup>, [Wanhe Yu](https://tongclass.ac.cn/author/wanhe-yu/)<sup>1</sup>, [Yuyang Yan](https://github.com/avx34/)<sup>1</sup>, [Xiaowei Chi](https://litwellchi.github.io/)<sup>5</sup>,  
[Yandong Guo](https://scholar.google.com/citations?user=fWDoWsQAAAAJ&hl=en)<sup>2</sup>, [Boxin Shi](https://camera.pku.edu.cn/)<sup>1</sup>, [Shanghang Zhang](https://www.shanghangzhang.com)<sup>1</sup>

<sup>1</sup>Peking University, <sup>2</sup>AI2Robotics, <sup>3</sup>Sun Yat-sen University, <sup>4</sup>Wuhan University, <sup>5</sup>HKUST

![Overview](assets/ZPD_Teaser.png)

</div>

This repository contains the RoboTwin implementation of **Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling**. It keeps a clean RoboTwin 2.0 codebase with a single policy folder, `policy/video2act`.

## Outline

- [Pipeline](#pipeline)
- [News](#news)
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
  - [1. T5 Text Embeddings](#1-t5-text-embeddings)
  - [2. HDF5 Files with Latent Precache](#2-hdf5-files-with-latent-precache)
  - [3. WAN Text Embeddings](#3-wan-text-embeddings)
- [Model](#model)
  - [Configuration files](#configuration-files)
- [Training](#training)
- [RoboTwin 2.0 Evaluation](#robotwin-20-evaluation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)
- [TODO](#todo)

## Pipeline

<div align="center">

![Pipeline](assets/pipeline.gif)

</div>

## News

- **[2026.06.24]** We prepare the RoboTwin Video2Act policy release.

## Overview

Video2Act is a dual-system video diffusion policy framework for robotic manipulation. This repository releases the RoboTwin 2.0 policy implementation in `policy/video2act`.

The policy learns action generation from robot observations, language instructions, and video-based spatio-motional representations. It is packaged as a RoboTwin 2.0 policy for training and benchmark evaluation.

This repository is intentionally lightweight. It does not include model checkpoints, video model weights, raw RoboTwin datasets, preprocessed trajectories, cached video latents, or experiment logs.

## Environment Setup

Install the following components before running Video2Act. We recommend using one merged environment instead of separate policy and video environments, so that PyTorch, CUDA, and DeepSpeed versions are not installed repeatedly.

1. RoboTwin 2.0

   Follow the official RoboTwin 2.0 installation guide and create the base RoboTwin conda environment first:

   - [RoboTwin 2.0 Install & Download](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

2. Video2Act requirements

   ```bash
   conda activate <robotwin_env>
   cd policy/video2act
   pip install -r requirements.txt
   ```

3. Required model weights

   Prepare these weights locally before data precaching, training, or evaluation:

   | Component | Hugging Face link | Used as |
   |-----------|-------------------|---------|
   | RDT-1B base policy checkpoint | [robotics-diffusion-transformer/rdt-1b](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | `pretrained_model_name_or_path` in `policy/video2act/model_config/*.yml` |
   | SigLIP vision encoder | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | `VISION_ENCODER_NAME` |
   | T5 language encoder | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `TEXT_ENCODER_NAME` |
   | Video model checkpoint | [XuWuLingYu/Wan2.2-5B-Robot](https://huggingface.co/XuWuLingYu/Wan2.2-5B-Robot) | `VIDEO2ACT_WAN_DIT_PATH` |
   | Video VAE checkpoint | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `VIDEO2ACT_WAN_VAE_PATH` |
   | Video text encoder and tokenizer | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `VIDEO2ACT_WAN_TEXT_ENCODER_PATH`, `VIDEO2ACT_WAN_TOKENIZER_PATH` |

   The RoboTwin RDT setup uses the same RDT-1B, SigLIP, and T5 checkpoints. See the RoboTwin RDT usage page for the baseline environment reference:

   - [RoboTwin RDT Usage](https://robotwin-platform.github.io/doc/usage/RDT.html)

   Example downloads:

   ```bash
   hf download robotics-diffusion-transformer/rdt-1b \
     --local-dir ./policy/weights/RDT/rdt-1b

   hf download google/siglip-so400m-patch14-384 \
     --local-dir ./policy/weights/RDT/siglip-so400m-patch14-384

   hf download google/t5-v1_1-xxl \
     --local-dir ./policy/weights/RDT/t5-v1_1-xxl

   hf download XuWuLingYu/Wan2.2-5B-Robot \
     --local-dir ./checkpoints/Wan2.2-5B-Robot

   hf download Wan-AI/Wan2.2-TI2V-5B \
     --local-dir ./checkpoints/Wan2.2-TI2V-5B
   ```

Set the required checkpoint paths before data precaching, training, or evaluation:

```bash
export VIDEO2ACT_CKPT_ROOT=/path/to/checkpoints

export VIDEO2ACT_WAN_DIT_PATH="$VIDEO2ACT_CKPT_ROOT/Wan2.2-5B-Robot/checkpoint.safetensors"
export VIDEO2ACT_WAN_VAE_PATH="$VIDEO2ACT_CKPT_ROOT/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export VIDEO2ACT_WAN_TEXT_ENCODER_PATH="$VIDEO2ACT_CKPT_ROOT/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
export VIDEO2ACT_WAN_TOKENIZER_PATH="$VIDEO2ACT_CKPT_ROOT/Wan2.2-TI2V-5B/google/umt5-xxl"
export VISION_ENCODER_NAME="$VIDEO2ACT_CKPT_ROOT/siglip-so400m-patch14-384"
export TEXT_ENCODER_NAME="$VIDEO2ACT_CKPT_ROOT/t5-v1_1-xxl"
```

In `policy/video2act/model_config/*.yml`, set `pretrained_model_name_or_path`
to the RDT-1B base checkpoint or a trained Video2Act checkpoint.

## Data Preparation

First collect or download RoboTwin 2.0 demonstrations following the RoboTwin documentation. The raw data is expected under:

```text
data/<task_name>/<task_config>/
  data/episode*.hdf5
  instructions/episode*.json
```

For the RoboTwin 20-task Video2Act setup, we prepared the WAN22 cached HDF5 dataset
used by the laplace-only ablation config. The planned Hugging Face dataset entry is:

- [jiayueru/Video2Act-RoboTwin-20Tasks-WAN22](https://huggingface.co/datasets/jiayueru/Video2Act-RoboTwin-20Tasks-WAN22)

Video2Act data preparation has three parts:

1. T5 text embeddings
2. HDF5 files with latent precache
3. WAN text embeddings

### 1. T5 Text Embeddings

Convert the official RoboTwin data to the RDT-format trajectory files used by Video2Act:

```bash
cd policy/video2act
bash process_data_rdt.sh <task_name> <task_config> <episode_num> <gpu_id>
```

Example:

```bash
bash process_data_rdt.sh click_bell aloha-agilex_clean_50 50 0
```

This wrapper calls `scripts/process_data.py` and follows the official RoboTwin RDT data conversion layout.

This creates:

```text
policy/video2act/processed_data/<task_name>-<task_config>-<episode_num>/
```

Each processed episode contains robot actions, qpos, three camera views, and T5 language embedding files under `episode_*/instructions/`.

If T5 text embeddings need to be regenerated for a model config, use:

```bash
python scripts/precompute_rdt_text_embeddings.py \
  --model_config_path model_config/video2act_example.yml \
  --config_path configs/video2act_template.yaml \
  --t5_path /path/to/t5-v1_1-xxl \
  --gpu 0
```

For many tasks, write a small batch wrapper around `process_data_rdt.sh`. The
wrapper should:

- extract RoboTwin task zip files if needed,
- skip tasks that are already complete,
- convert only missing episodes,
- write processed task directories under `policy/video2act/processed_data/`.

### 2. HDF5 Files with Latent Precache

Video2Act trains with cached video latents. After data preparation, call
`policy/video2act/scripts/precache_wan22_latents.py` for each processed task
directory. For many tasks, put the command below in your own batch wrapper and
iterate over the processed task directories. The precache step only needs to
generate video latent keys; text embeddings are not required in this stage.

The default command for one processed task is:

```bash
cd policy/video2act
python scripts/precache_wan22_latents.py \
  --mode both \
  --src_processed_dir processed_data/<task_name>-<task_config>-<episode_num> \
  --dst_processed_dir processed_data/<task_name>-<task_config>-<episode_num>_wan22_832x480 \
  --vae_path "$VIDEO2ACT_WAN_VAE_PATH" \
  --device cuda:0 \
  --dtype bf16 \
  --camera_key cam_high \
  --height 480 \
  --width 832 \
  --motion_height 224 \
  --motion_width 224 \
  --motion_frames 61
```

The latent-precache step writes video latent groups into the processed HDF5 files:

```text
observations/wan22_image_latents/head
observations/wan22_image_latents_motion/head
```

For multi-GPU or multi-machine precaching, split the processed episodes with:

```bash
--num_shards <N> --shard_index <i>
```

The training config should point `data_path` to the cached processed directories, for example directories ending with `_wan22_832x480`.

### 3. WAN Text Embeddings

Optionally precompute the task-level WAN UMT5 instruction embeddings after the
processed data directories are available:

```bash
cd policy/video2act
python scripts/precompute_wan_text_embeddings.py \
  --model_config_path model_config/video2act_example.yml \
  --desc_type seen \
  --gpu 0
```

This writes:

```text
<task_dir>/text_embeddings/seen_0000.pkl
```

For evaluation with unseen descriptions, run the same script with
`--desc_type unseen`; it writes `<task_dir>/unseen_text_embeddings/`.

If these files are missing, the loader prints a warning and falls back to zero
WAN text conditioning.

## Model

Video2Act receives robot state, language instruction, and multi-view image observations from RoboTwin. The model builds a video-aware representation of the scene and predicts a chunk of bimanual robot actions with a diffusion policy objective.

The public entry points are:

```text
policy/video2act/model.py
policy/video2act/deploy_policy.py
policy/video2act/configs/                # architecture configs
policy/video2act/model_config/           # per-experiment training configs
```

### Configuration files

Two complementary config layers are kept, each as a `template` (placeholders
to copy + edit) and an `example` (concrete reference values):

| layer | template | example |
|---|---|---|
| **`model_config/*.yml`** — per-experiment (data paths, hparams, GPUs) | `model_config/video2act_template.yml` | `model_config/video2act_example.yml` |
| **`configs/*.yaml`** — model architecture + WAN integration | `configs/video2act_template.yaml` | `configs/video2act_example.yaml` |

- **Template files** are stripped to placeholders (`/path/to/...`) and use
  `${VIDEO2ACT_WAN_*}` env vars for WAN inference weights — start here for a
  fresh run.
- **Example files** are concrete reference configs (annotated, multi-task,
  hard-coded WAN paths) — read them to understand non-default fields.

`finetune.sh` and `eval.sh` default to `model_config/video2act_example.yml`,
which references `configs/video2act_example.yaml`. Override with `--config`
flags or `MODEL_CFG=...` env to use the templates instead.

Fields you typically edit in the per-experiment YAML before training:

- `data_path`: processed cached-data directories.
- `checkpoint_path`: where training checkpoints will be saved.
- `pretrained_model_name_or_path`: RDT-1B base checkpoint or previous Video2Act checkpoint.
- `config_path`: which architecture YAML to load (template or example).

## Training

From `policy/video2act`:

```bash
cd policy/video2act
GPUS_LIST="0 1 2 3 4 5 6 7" bash finetune.sh video2act_example
```

`finetune.sh` reads `model_config/<name>.yml`, queues runs across the GPUs in
`GPUS_LIST`, and writes per-config logs under `logs/finetune_<timestamp>/`.
For a new sweep, copy `model_config/video2act_template.yml`, fill in
`data_path` / `checkpoint_path` / `pretrained_model_name_or_path`, then pass
its filename (without `.yml`) to `finetune.sh`.

For multi-config queueing on the same GPUs:

```bash
CONFIGS="cfg_a cfg_b cfg_c" GPUS_LIST="0 1 2 3" bash finetune.sh
```

## RoboTwin 2.0 Evaluation

Video2Act follows the standard RoboTwin policy deployment interface, with the
addition of WAN-encoder env vars and a `model_config_path` flag for
fixed-embedding eval:

```text
policy/video2act/deploy_policy.py
policy/video2act/deploy_policy.yml
policy/video2act/eval.sh                    # canonical 6-arg entry
```

Before launching, export the four WAN inference weights (referenced as
`${VIDEO2ACT_WAN_*}` in `configs/video2act_template.yaml`):

```bash
export VIDEO2ACT_WAN_DIT_PATH=/path/to/Wan2.2-5B-Robot/checkpoint.safetensors
export VIDEO2ACT_WAN_VAE_PATH=/path/to/wan-t2v-5b/Wan2.2_VAE.pth
export VIDEO2ACT_WAN_TEXT_ENCODER_PATH=/path/to/wan-t2v-5b/models_t5_umt5-xxl-enc-bf16.pth
export VIDEO2ACT_WAN_TOKENIZER_PATH=/path/to/google/umt5-xxl
```

Then evaluate one task:

```bash
bash policy/video2act/eval.sh <task_name> <task_config> <ckpt_dir> <checkpoint_id> <seed> <gpu_id>
```

Example:

```bash
bash policy/video2act/eval.sh click_bell demo_clean /path/to/checkpoint-20000 20000 0 0
```

`<ckpt_dir>` must contain `pytorch_model.bin` (single-card) or
`pytorch_model/mp_rank_00_model_states.pt` (DeepSpeed). `<checkpoint_id>`
is only a label for the output dir.

`eval.sh` auto-loads fixed lang / WAN text embeddings from
`processed_data/<task>-aloha-agilex_clean_50-50_wan22_832x480/` when present;
override with `MODEL_CFG`, `PROCESSED_DATA_ROOT`, `FIXED_LANG_EMBED` env vars.

**Slow-fast WAN cache (eval-only).** WAN feature extraction dominates inference
cost. Set `CACHE_RATIO=N` to recompute WAN once every N policy calls and reuse
the cached `adapted_hidden_states` in between (e.g. 1:8 ≈ 8× WAN speedup with
negligible accuracy loss for short horizons). YAML defaults live in
`configs/video2act_*.yaml` (`framepack.enable_cache`, `framepack.cache_ratio`,
default `1:8`); the env var overrides both. `CACHE_RATIO=1` disables the cache.

```bash
CACHE_RATIO=8 bash policy/video2act/eval.sh click_bell demo_clean /path/to/checkpoint-20000 20000 0 0
```

**Requirements** (`eval.sh` aborts pre-flight if any is missing):

- Conda env `RoboTwin` (SAPIEN + mplib + curobo); auto-activated, override with `POLICY_CONDA_ENV`.
- `Lightewm/` next to this repo.
- `assets/{objects,embodiments,background_texture}/` — symlink from a full RoboTwin 2.0 dataset.
- ≥ 30 GB free GPU memory.

For large-scale evaluation, wrap `eval.sh` with your task list and GPU
scheduler.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{jia2025video2actdualsystemvideodiffusion,
      title={Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling}, 
      author={Yueru Jia and Jiaming Liu and Shengbang Liu and Rui Zhou and Wanhe Yu and Yuyang Yan and Xiaowei Chi and Yandong Guo and Boxin Shi and Shanghang Zhang},
      year={2025},
      eprint={2512.03044},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.03044}, 
}
```

## Acknowledgement

This implementation is built on [RoboTwin](https://github.com/robotwin-Platform/RoboTwin) and [LightEWM](https://github.com/XuWuLingYu/LightEWM). We thank the open-source robotics and video generation communities for research infrastructure.

## License

This project will be released under the MIT License.

## TODO

- We update our model from Hunyuan to WAN for better efficiency and performance. The updated paper will release soon.

- [ ] Upload latent-precache dataset
- [ ] Upload eval checkpoint
- [ ] Add benchmark score table
- [ ] Link updated paper
