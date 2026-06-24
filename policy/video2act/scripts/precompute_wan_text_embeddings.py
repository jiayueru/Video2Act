#!/usr/bin/env python3
"""Precompute WAN UMT5 text embeddings for Video2Act."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm


VIDEO2ACT_ROOT = Path(__file__).resolve().parents[1]
if str(VIDEO2ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO2ACT_ROOT))

from config_utils import load_yaml_with_env  # noqa: E402
from models.framepack_lightweight_encoder import FramepackLightweightEncoder  # noqa: E402


def resolve_path(path, base=VIDEO2ACT_ROOT):
    path = Path(os.path.expandvars(str(path))).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_task_dirs(model_config_path):
    model_config = load_yaml_with_env(model_config_path)
    data_path = model_config["data_path"]
    task_dirs = data_path if isinstance(data_path, list) else [data_path]
    return [resolve_path(path) for path in task_dirs]


def read_instruction_texts(task_dir, desc_type):
    instr_root = task_dir / "instructions"
    json_paths = []
    if instr_root.is_dir():
        json_paths.extend(sorted(instr_root.glob("episode*.json")))
    # Fallback: per-episode instructions/*.json (post-latent-cache layout)
    if not json_paths:
        json_paths = sorted(task_dir.glob("episode_*/instructions/*.json"))

    if not json_paths:
        return [], [str(instr_root)]

    texts = []
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        values = payload.get(desc_type, [])
        if isinstance(values, str):
            values = [values]
        texts.extend(text for text in values if text)

    return list(dict.fromkeys(texts)), []


def collect_jobs(task_dirs, desc_type, overwrite):
    folder = "unseen_text_embeddings" if desc_type == "unseen" else "text_embeddings"
    prefix = "unseen" if desc_type == "unseen" else "seen"
    jobs = []
    missing = []
    skipped = 0

    for task_dir in task_dirs:
        texts, task_missing = read_instruction_texts(task_dir, desc_type)
        missing.extend(task_missing)
        out_dir = task_dir / folder
        for idx, text in enumerate(texts):
            out_path = out_dir / f"{prefix}_{idx:04d}.pkl"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            jobs.append((text, out_path))

    return jobs, missing, skipped


@torch.inference_mode()
def write_embeddings(jobs, encoder):
    written = 0
    cache = {}
    for text, out_path in tqdm(jobs, desc="encoding"):
        if text not in cache:
            ctx, _ = encoder.encode_text(text)
            cache[text] = ctx
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache[text], out_path)
        written += 1
    return written, len(cache)


def run_one_desc_type(args, desc_type, encoder):
    task_dirs = load_task_dirs(resolve_path(args.model_config_path))
    jobs, missing, skipped = collect_jobs(task_dirs, desc_type, args.overwrite)

    print(f"[INFO] desc_type    : {desc_type}")
    print(f"[INFO] task_dirs    : {len(task_dirs)}")
    print(f"[INFO] save jobs    : {len(jobs)}")
    print(f"[INFO] skipped      : {skipped}")
    if missing:
        print("[WARN] Missing instruction roots:")
        for path in missing:
            print(f"  {path}")
    if not jobs:
        return 0

    written, unique_texts = write_embeddings(jobs, encoder)
    print(f"[DONE] desc_type={desc_type} written={written} unique_texts={unique_texts}")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        default=str(VIDEO2ACT_ROOT / "model_config/video2act_example.yml"),
        help="Model config whose data_path points to processed task directories.",
    )
    parser.add_argument(
        "--text_encoder_path",
        default=os.environ.get("VIDEO2ACT_WAN_TEXT_ENCODER_PATH", ""),
        help="Wan2.2 UMT5 text encoder checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=os.environ.get("VIDEO2ACT_WAN_TOKENIZER_PATH", ""),
        help="Wan2.2 UMT5 tokenizer directory.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--desc_type", choices=["seen", "unseen", "all"], default="seen")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.text_encoder_path:
        raise ValueError("Set --text_encoder_path or VIDEO2ACT_WAN_TEXT_ENCODER_PATH.")
    if not args.tokenizer_path:
        raise ValueError("Set --tokenizer_path or VIDEO2ACT_WAN_TOKENIZER_PATH.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] model_config : {resolve_path(args.model_config_path)}")
    print(f"[INFO] text_encoder : {resolve_path(args.text_encoder_path)}")
    print(f"[INFO] tokenizer    : {resolve_path(args.tokenizer_path)}")
    print(f"[INFO] device       : {device}")

    encoder = FramepackLightweightEncoder(
        text_encoder1_path=str(resolve_path(args.text_encoder_path)),
        tokenizer_path=str(resolve_path(args.tokenizer_path)),
        device=device,
        load_vae=False,
        load_text_encoders=True,
    )

    desc_types = ["seen", "unseen"] if args.desc_type == "all" else [args.desc_type]
    total = 0
    for desc_type in desc_types:
        total += run_one_desc_type(args, desc_type, encoder)
    if total == 0:
        raise RuntimeError("No WAN text embeddings were written.")


if __name__ == "__main__":
    main()
