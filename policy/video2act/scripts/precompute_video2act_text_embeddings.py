#!/usr/bin/env python3
"""Precompute RDT T5 text embeddings into episode instructions directories.

This only creates the Video2Act-side `.pt` files consumed by `--precomp_lang_embed`:

    processed_data/TASK/episode_N/instructions/lang_embed_K.pt

It does not touch HDF5 files, WAN VAE latents, or WAN UMT5 `.pkl` embeddings.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


VIDEO2ACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VIDEO2ACT_ROOT.parents[2]
DEFAULT_T5_PATH = REPO_ROOT / "Ckpt/weights/t5-v1_1-xxl"

if str(VIDEO2ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO2ACT_ROOT))

from models.multimodal_encoder.t5_encoder import T5Embedder  # noqa: E402


def resolve_path(path, base=VIDEO2ACT_ROOT):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def load_task_dirs(model_config_path):
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    data_path = model_config["data_path"]
    task_dirs = data_path if isinstance(data_path, list) else [data_path]
    return [resolve_path(p) for p in task_dirs]


def load_base_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_jobs(task_dirs, desc_type):
    """Collect embedding save jobs and unique texts.

    Returns:
        jobs: list[(text, target_path)]
        unique_texts: sorted list[str]
    """
    jobs = []
    unique_texts = set()
    missing = []

    for task_dir in task_dirs:
        instr_root = task_dir / "instructions"
        if not instr_root.is_dir():
            missing.append(str(instr_root))
            continue

        for json_path in sorted(instr_root.glob("episode*.json")):
            stem = json_path.stem
            if not stem.startswith("episode"):
                continue
            episode_id = stem[len("episode") :]
            target_dir = task_dir / f"episode_{episode_id}" / "instructions"
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = data.get(desc_type, [])
            for idx, text in enumerate(texts):
                if not text:
                    continue
                target_path = target_dir / f"lang_embed_{idx}.pt"
                jobs.append((text, target_path))
                unique_texts.add(text)

    if missing:
        print("[WARN] Missing instruction roots:")
        for path in missing:
            print(f"  {path}")

    return jobs, sorted(unique_texts)


@torch.inference_mode()
def encode_unique_texts(texts, embedder, batch_size):
    tokenizer = embedder.tokenizer
    text_encoder = embedder.model
    device = embedder.device

    encoded = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attn_mask = tokenized["attention_mask"].to(device)
        text_embeds = text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )["last_hidden_state"].detach().cpu()
        attn_mask_cpu = attn_mask.cpu().bool()

        for text, emb, mask in zip(batch, text_embeds, attn_mask_cpu):
            encoded[text] = emb[mask].contiguous()

    return encoded


def write_jobs(jobs, encoded, overwrite):
    written = 0
    skipped = 0
    for text, target_path in tqdm(jobs, desc="writing"):
        if target_path.exists() and not overwrite:
            skipped += 1
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded[text], target_path)
        written += 1
    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        default=str(VIDEO2ACT_ROOT / "model_config/video2act_example.yml"),
    )
    parser.add_argument(
        "--config_path",
        default=str(VIDEO2ACT_ROOT / "configs/video2act_template.yaml"),
    )
    parser.add_argument("--t5_path", default=str(DEFAULT_T5_PATH))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--desc_type", choices=["seen", "unseen"], default="seen")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--offload_dir",
        default=None,
        help="Optional accelerate offload dir for low-memory GPUs.",
    )
    args = parser.parse_args()

    model_config_path = resolve_path(args.model_config_path)
    config_path = resolve_path(args.config_path)
    t5_path = resolve_path(args.t5_path, REPO_ROOT)
    base_config = load_base_config(config_path)
    task_dirs = load_task_dirs(model_config_path)
    jobs, unique_texts = collect_jobs(task_dirs, args.desc_type)

    print(f"[INFO] model_config : {model_config_path}")
    print(f"[INFO] config       : {config_path}")
    print(f"[INFO] t5_path      : {t5_path}")
    print(f"[INFO] task_dirs    : {len(task_dirs)}")
    print(f"[INFO] save jobs    : {len(jobs)}")
    print(f"[INFO] unique texts : {len(unique_texts)}")
    if not jobs:
        raise RuntimeError("No instruction embedding jobs found.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device       : {device}")

    embedder = T5Embedder(
        from_pretrained=str(t5_path),
        model_max_length=base_config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir,
        local_files_only=True,
    )

    encoded = encode_unique_texts(unique_texts, embedder, args.batch_size)
    written, skipped = write_jobs(jobs, encoded, args.overwrite)
    print(f"[DONE] written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
