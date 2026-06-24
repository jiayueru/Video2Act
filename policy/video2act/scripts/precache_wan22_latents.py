import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[4]
VIDEO2ACT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIGHTEWM_ROOT = None
DEFAULT_VAE_PATH = VIDEO2ACT_ROOT / "checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
if str(VIDEO2ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO2ACT_ROOT))


def resolve_path(path, base=VIDEO2ACT_ROOT):
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def parse_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_wan22_vae(args, device, dtype):
    vae_path = resolve_path(args.vae_path) if args.vae_path else None

    if vae_path is not None and vae_path.exists():
        if args.lightewm_root:
            lightewm_root = resolve_path(args.lightewm_root, REPO_ROOT)
            if str(lightewm_root) not in sys.path:
                sys.path.insert(0, str(lightewm_root))

            from lightewm.model.wan.pipeline_ti2v_5b import WanTI2V5BPipeline
            from lightewm.utils.loader.config import ModelConfig

            pipe = WanTI2V5BPipeline.from_pretrained(
                torch_dtype=dtype,
                device=device,
                model_configs=[ModelConfig(path=str(vae_path))],
                tokenizer_config=None,
                redirect_common_files=True,
            )
            if pipe.vae is None:
                raise RuntimeError(f"Wan2.2 VAE was not loaded from {vae_path}")
            pipe.vae.eval()
            return pipe.vae

        from third_party.lightewm_minimal import load_wan22_vae as load_minimal_wan22_vae

        return load_minimal_wan22_vae(vae_path, device=device, dtype=dtype)
    elif args.allow_download:
        if not args.lightewm_root:
            raise ValueError("--allow_download requires --lightewm_root pointing to a full LightEWM checkout.")
        lightewm_root = resolve_path(args.lightewm_root, REPO_ROOT)
        if str(lightewm_root) not in sys.path:
            sys.path.insert(0, str(lightewm_root))

        from lightewm.model.wan.pipeline_ti2v_5b import WanTI2V5BPipeline
        from lightewm.utils.loader.config import ModelConfig

        model_config = ModelConfig(
            model_id=args.vae_model_id,
            origin_file_pattern=args.vae_origin_file_pattern,
            download_source=args.download_source,
            local_model_path=args.model_base_path,
        )
    else:
        raise FileNotFoundError(
            f"Wan2.2 VAE path does not exist: {vae_path}. "
            "Pass --allow_download to fetch it with ModelConfig, or set --vae_path."
        )

    pipe = WanTI2V5BPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[model_config],
        tokenizer_config=None,
        redirect_common_files=True,
    )
    if pipe.vae is None:
        raise RuntimeError(f"Wan2.2 VAE was not loaded from {vae_path}")
    pipe.vae.eval()
    return pipe.vae


def preprocess_image(raw_bytes, width, height):
    image_bytes = bytes(raw_bytes).rstrip(b"\0")
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG bytes from HDF5 image dataset")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(1)


def build_motion_video(images_ds, step_idx, num_history, width, height, dtype):
    """Build a (C, T=num_history, H, W) video ending at step_idx, with replication padding."""
    total = images_ds.shape[0]
    indices = [max(0, step_idx - num_history + 1 + i) for i in range(num_history)]
    frames = []
    for idx in indices:
        raw = images_ds[min(idx, total - 1)]
        img_bytes = bytes(raw).rstrip(b"\0")
        bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)   # (C, H, W)
        frames.append(t)
    # stack → (C, T, H, W), normalize to [-1, 1]
    video = torch.stack(frames, dim=1).to(dtype=dtype)   # (C, T, H, W)
    video = video * 2.0 - 1.0
    return video


def preload_motion_frames(images_ds, width, height):
    """Decode and resize all episode frames once for motion sliding-window reuse."""
    frames = []
    for idx in range(images_ds.shape[0]):
        raw = images_ds[idx]
        img_bytes = bytes(raw).rstrip(b"\0")
        bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode JPEG bytes from HDF5 image dataset at index {idx}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    return frames


def build_motion_video_from_frames(frames, step_idx, num_history, dtype):
    """Build a normalized (C, T, H, W) motion window from predecoded frames."""
    total = len(frames)
    indices = [max(0, step_idx - num_history + 1 + i) for i in range(num_history)]
    video = torch.stack([frames[min(idx, total - 1)] for idx in indices], dim=1)
    video = video.to(dtype=dtype)
    return video * 2.0 - 1.0


def find_hdf5_files(processed_dir):
    return sorted(Path(processed_dir).glob("**/*.hdf5"))


def read_hdf5_list(list_file, processed_dir):
    paths = []
    processed_dir = Path(processed_dir).resolve()
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            path = Path(item)
            if not path.is_absolute():
                path = processed_dir / path
            paths.append(path.resolve())
    return paths


def prepare_dst(src_processed_dir, dst_processed_dir, inplace, reuse_dst):
    src_processed_dir = resolve_path(src_processed_dir)
    if inplace:
        return src_processed_dir

    dst_processed_dir = resolve_path(dst_processed_dir)
    if dst_processed_dir.exists():
        if not reuse_dst:
            raise FileExistsError(
                f"{dst_processed_dir} already exists. Use --reuse_dst to append/cache there."
            )
        return dst_processed_dir

    shutil.copytree(src_processed_dir, dst_processed_dir)
    return dst_processed_dir


def latent_dataset_is_complete(dataset, expected_shape, require_complete_attr=False):
    if dataset.shape != expected_shape:
        return False
    complete_attr = dataset.attrs.get("cache_complete", None)
    if complete_attr is not None:
        return bool(complete_attr)
    return not require_complete_attr


def ensure_latent_dataset(obs, group_name, view_name, num_frames, latent_shape, overwrite, require_complete_attr=False):
    group = obs.require_group(group_name)
    expected_shape = (num_frames, 1, *latent_shape)
    if view_name in group:
        dataset = group[view_name]
        if not overwrite and latent_dataset_is_complete(dataset, expected_shape, require_complete_attr):
            return dataset, True
        if dataset.shape == expected_shape and not overwrite and (
            dataset.attrs.get("cache_complete", None) is not None or require_complete_attr
        ):
            del group[view_name]
        elif dataset.shape == expected_shape and not overwrite:
            return dataset, True
        elif overwrite:
            del group[view_name]
        else:
            raise ValueError(
                f"Existing dataset {group_name}/{view_name} has shape {dataset.shape}, "
                f"expected {expected_shape}. Use --overwrite to replace it."
            )

    if view_name in group:
        dataset = group[view_name]
        if dataset.shape == expected_shape and not overwrite:
            return dataset, True
        if not overwrite:
            raise ValueError(
                f"Existing dataset {group_name}/{view_name} has shape {dataset.shape}, "
                f"expected {expected_shape}. Use --overwrite to replace it."
            )
        del group[view_name]

    dataset = group.create_dataset(
        view_name,
        shape=(num_frames, 1, *latent_shape),
        dtype=np.float16,
        chunks=(1, 1, *latent_shape),
        compression="lzf",
    )
    dataset.attrs["cache_complete"] = 0
    return dataset, False


def latent_dataset_complete(hdf5_path, args, group_name, latent_shape):
    key = f"observations/{group_name}/{args.view_name}"
    try:
        with h5py.File(hdf5_path, "r") as f:
            num_frames = f["observations"]["images"][args.camera_key].shape[0]
            expected_shape = (num_frames, 1, *latent_shape)
            return key in f and latent_dataset_is_complete(
                f[key],
                expected_shape,
                args.require_complete_attr,
            )
    except Exception:
        return False


def encode_videos_with_vae(videos, vae, args):
    if args.vae_encode_backend == "batch" and not args.tiled:
        batch = torch.stack(videos, dim=0).to(device=args.device, dtype=parse_dtype(args.dtype))
        return vae.model.encode(batch, vae.scale)

    return vae.encode(
        videos,
        device=args.device,
        tiled=args.tiled,
        tile_size=(args.tile_size, args.tile_size),
        tile_stride=(args.tile_stride, args.tile_stride),
    )


def episode_needs_cache(hdf5_path, args):
    raw_shape = (48, 1, args.height // 16, args.width // 16)
    motion_shape = (
        48,
        (args.motion_frames - 1) // 4 + 1,
        args.motion_height // 16,
        args.motion_width // 16,
    )

    if args.mode == "raw":
        group_name = args.group_name or args.raw_group_name
        return args.overwrite or not latent_dataset_complete(hdf5_path, args, group_name, raw_shape)
    if args.mode == "motion":
        group_name = args.group_name or args.motion_group_name
        return args.overwrite or not latent_dataset_complete(hdf5_path, args, group_name, motion_shape)

    needs_raw = (
        args.overwrite
        or args.overwrite_raw
        or not latent_dataset_complete(hdf5_path, args, args.raw_group_name, raw_shape)
    )
    needs_motion = (
        args.overwrite
        or args.overwrite_motion
        or not latent_dataset_complete(hdf5_path, args, args.motion_group_name, motion_shape)
    )
    return needs_raw or needs_motion


@torch.inference_mode()
def cache_episode_raw(hdf5_path, vae, args):
    """Mode=raw: encode each frame independently → (N, 1, 48, 1, H/16, W/16)."""
    with h5py.File(hdf5_path, "a") as f:
        obs = f["observations"]
        images = obs["images"][args.camera_key]
        num_frames = images.shape[0]
        latent_h = args.height // 16
        latent_w = args.width // 16
        latent_shape = (48, 1, latent_h, latent_w)
        latent_ds, already_cached = ensure_latent_dataset(
            obs,
            args.group_name,
            args.view_name,
            num_frames,
            latent_shape,
            args.overwrite,
            args.require_complete_attr,
        )
        if already_cached:
            return "skip"

        iterator = range(0, num_frames, args.batch_size)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=hdf5_path.name, leave=False)

        for start in iterator:
            end = min(num_frames, start + args.batch_size)
            videos = [
                preprocess_image(images[idx], args.width, args.height).to(dtype=parse_dtype(args.dtype))
                for idx in range(start, end)
            ]
            latents = encode_videos_with_vae(videos, vae, args)
            latent_ds[start:end, 0] = latents.detach().cpu().to(torch.float16).numpy()

        latent_ds.attrs["source_camera"] = args.camera_key
        latent_ds.attrs["image_width"] = args.width
        latent_ds.attrs["image_height"] = args.height
        latent_ds.attrs["latent_format"] = "Wan2.2_VAE38_raw [N,1,48,1,H/16,W/16]"
        latent_ds.attrs["vae_encode_backend"] = args.vae_encode_backend
        latent_ds.attrs["cache_complete"] = 1
        f.flush()
        return "write"


@torch.inference_mode()
def cache_episode_motion(hdf5_path, vae, args):
    """Mode=motion: per-step sliding window of motion_frames → (N, 1, 48, T_lat, H/16, W/16).

    For motion_frames=61 frames and Wan2.2 temporal stride=4:
        T_lat = (61 - 1) // 4 + 1 = 16
    Stores in group_name/view_name with shape (N, 1, 48, 16, H_m/16, W_m/16).
    """
    dtype = parse_dtype(args.dtype)
    latent_h = args.motion_height // 16
    latent_w = args.motion_width // 16
    # T_lat = (motion_frames - 1) // 4 + 1  (Wan2.2 temporal compression ×4)
    t_lat = (args.motion_frames - 1) // 4 + 1
    latent_shape = (48, t_lat, latent_h, latent_w)

    with h5py.File(hdf5_path, "a") as f:
        obs = f["observations"]
        images = obs["images"][args.camera_key]
        num_frames = images.shape[0]

        latent_ds, already_cached = ensure_latent_dataset(
            obs,
            args.group_name,
            args.view_name,
            num_frames,
            latent_shape,
            args.overwrite,
            args.require_complete_attr,
        )
        if already_cached:
            return "skip"

        decoded_frames = preload_motion_frames(images, args.motion_width, args.motion_height)
        motion_batch_size = args.motion_batch_size or args.batch_size
        iterator = range(0, num_frames, motion_batch_size)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=hdf5_path.name, leave=False)

        for start in iterator:
            end = min(num_frames, start + motion_batch_size)
            videos = [
                build_motion_video_from_frames(decoded_frames, step_idx, args.motion_frames, dtype)
                for step_idx in range(start, end)
            ]  # each: (C=3, T=61, H=224, W=224)
            latents = encode_videos_with_vae(videos, vae, args)  # (B, 48, T_lat, H_m/16, W_m/16)
            latents = latents.detach().cpu().to(torch.float16)
            if latents.ndim == 4:
                latents = latents.unsqueeze(0)
            latent_ds[start:end, 0] = latents.numpy()

        latent_ds.attrs["source_camera"] = args.camera_key
        latent_ds.attrs["motion_height"] = args.motion_height
        latent_ds.attrs["motion_width"] = args.motion_width
        latent_ds.attrs["motion_frames"] = args.motion_frames
        latent_ds.attrs["latent_format"] = f"Wan2.2_VAE38_motion [N,1,48,{t_lat},{latent_h},{latent_w}]"
        latent_ds.attrs["vae_encode_backend"] = args.vae_encode_backend
        latent_ds.attrs["cache_complete"] = 1
        f.flush()
        return "write"


# keep old name as alias for backward compatibility
cache_episode = cache_episode_raw


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Precache Wan2.2 VAE latents into HDF5.\n"
            "  mode=raw   : per-frame encode at (height × width), "
            "stores wan22_image_latents/head   (N,1,48,1,H/16,W/16)\n"
            "  mode=motion: 61-frame sliding-window encode at (motion_height × motion_width), "
            "stores wan22_image_latents_motion/head (N,1,48,16,H_m/16,W_m/16)"
        )
    )
    parser.add_argument("--mode", choices=["raw", "motion", "both"], default="raw",
                        help="raw=per-frame head cam; motion=61-frame sliding window; both=load VAE once and cache both")
    parser.add_argument("--src_processed_dir", required=True)
    parser.add_argument("--dst_processed_dir", default=None)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--reuse_dst", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lightewm_root", default=DEFAULT_LIGHTEWM_ROOT,
                        help="Optional external full LightEWM checkout. By default, local minimal Wan2.2 VAE code is used.")
    parser.add_argument("--vae_path", default=str(DEFAULT_VAE_PATH))
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--vae_model_id", default="DiffSynth-Studio/Wan-Series-Converted-Safetensors")
    parser.add_argument("--vae_origin_file_pattern", default="Wan2.2_VAE.safetensors")
    parser.add_argument("--download_source", choices=["modelscope", "huggingface"], default="huggingface")
    parser.add_argument("--model_base_path", default=str(VIDEO2ACT_ROOT / "checkpoints"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    # raw-stream image size
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--batch_size", type=int, default=1)
    # motion-stream settings
    parser.add_argument("--motion_height", type=int, default=224,
                        help="Resize height for motion-stream frames (mode=motion)")
    parser.add_argument("--motion_width", type=int, default=224,
                        help="Resize width for motion-stream frames (mode=motion)")
    parser.add_argument("--motion_frames", type=int, default=61,
                        help="History window length for motion stream (mode=motion)")
    parser.add_argument("--motion_batch_size", type=int, default=None,
                        help="Batch size for motion VAE encode. Defaults to --batch_size.")
    parser.add_argument("--vae_encode_backend", choices=["serial", "batch"], default="serial",
                        help="serial uses LightEWM vae.encode(list); batch stacks videos and calls vae.model.encode once.")
    parser.add_argument("--require_complete_attr", action="store_true",
                        help="Require cache_complete=1 before treating a latent dataset as already cached.")
    # HDF5 target keys
    parser.add_argument("--camera_key", default="cam_high")
    parser.add_argument("--group_name", default=None,
                        help="HDF5 group under observations/ (default: wan22_image_latents for raw, "
                             "wan22_image_latents_motion for motion)")
    parser.add_argument("--raw_group_name", default="wan22_image_latents")
    parser.add_argument("--motion_group_name", default="wan22_image_latents_motion")
    parser.add_argument("--view_name", default="head")
    parser.add_argument("--hdf5_list_file", default=None,
                        help="Optional newline-delimited HDF5 paths to process instead of scanning src/dst.")
    parser.add_argument("--limit_episodes", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite_raw", action="store_true",
                        help="Only used with --mode both; replace raw latent datasets.")
    parser.add_argument("--overwrite_motion", action="store_true",
                        help="Only used with --mode both; replace motion latent datasets.")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=34)
    parser.add_argument("--tile_stride", type=int, default=18)
    return parser


def main():
    args = build_parser().parse_args()

    if args.mode in ("raw", "both"):
        if args.height % 32 != 0 or args.width % 32 != 0:
            raise ValueError("Wan2.2 raw image size must be divisible by 32.")
    if args.mode in ("motion", "both"):
        if args.motion_height % 16 != 0 or args.motion_width % 16 != 0:
            raise ValueError("Motion image size must be divisible by 16.")
    if args.motion_batch_size is not None and args.motion_batch_size < 1:
        raise ValueError("--motion_batch_size must be >= 1")
    if args.vae_encode_backend == "batch" and args.tiled:
        raise ValueError("--vae_encode_backend batch is not compatible with --tiled")

    # Set default group_name based on single mode. In mode=both the per-stream
    # group names are selected inside cache_episode_both.
    if args.group_name is None and args.mode != "both":
        args.group_name = args.raw_group_name if args.mode == "raw" else args.motion_group_name

    if args.mode == "raw":
        if args.dst_processed_dir is None and not args.inplace:
            src = resolve_path(args.src_processed_dir)
            args.dst_processed_dir = str(src.with_name(src.name + "_wan22_832x480"))
        cache_fn = cache_episode_raw
    elif args.mode == "motion":
        if args.dst_processed_dir is None and not args.inplace:
            src = resolve_path(args.src_processed_dir)
            args.dst_processed_dir = str(src.with_name(src.name + "_wan22_832x480"))
        cache_fn = cache_episode_motion
    else:
        if args.dst_processed_dir is None and not args.inplace:
            src = resolve_path(args.src_processed_dir)
            args.dst_processed_dir = str(src.with_name(src.name + "_wan22_832x480"))
        cache_fn = None

    dst_processed_dir = prepare_dst(
        args.src_processed_dir,
        args.dst_processed_dir,
        args.inplace,
        args.reuse_dst,
    )
    if args.hdf5_list_file:
        hdf5_files = read_hdf5_list(args.hdf5_list_file, dst_processed_dir)
    else:
        hdf5_files = find_hdf5_files(dst_processed_dir)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    hdf5_files = [
        path for idx, path in enumerate(hdf5_files)
        if idx % args.num_shards == args.shard_index
    ]
    if args.limit_episodes is not None:
        hdf5_files = hdf5_files[: args.limit_episodes]
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found under {dst_processed_dir}")

    total_hdf5_files = len(hdf5_files)
    hdf5_files = [path for path in hdf5_files if episode_needs_cache(path, args)]
    pre_skipped = total_hdf5_files - len(hdf5_files)
    if not hdf5_files:
        print(
            "Wan2.2 latent cache complete: "
            f"all {total_hdf5_files} episode(s) already cached, dst={dst_processed_dir}"
        )
        return
    if pre_skipped:
        print(f"Skipping {pre_skipped}/{total_hdf5_files} already-cached episode(s) before loading VAE.")

    vae = load_wan22_vae(args, args.device, parse_dtype(args.dtype))

    written = 0
    skipped = 0
    written_raw = 0
    skipped_raw = 0
    written_motion = 0
    skipped_motion = 0
    iterator = tqdm(hdf5_files, desc="episodes") if tqdm is not None else hdf5_files
    for hdf5_path in iterator:
        if args.mode == "both":
            original_group_name = args.group_name
            original_overwrite = args.overwrite

            args.group_name = args.raw_group_name
            args.overwrite = original_overwrite or args.overwrite_raw
            status_raw = cache_episode_raw(hdf5_path, vae, args)

            args.group_name = args.motion_group_name
            args.overwrite = original_overwrite or args.overwrite_motion
            status_motion = cache_episode_motion(hdf5_path, vae, args)

            args.group_name = original_group_name
            args.overwrite = original_overwrite

            written_raw += int(status_raw == "write")
            skipped_raw += int(status_raw == "skip")
            written_motion += int(status_motion == "write")
            skipped_motion += int(status_motion == "skip")
        else:
            status = cache_fn(hdf5_path, vae, args)
            written += int(status == "write")
            skipped += int(status == "skip")

    if args.mode == "both":
        print(
            "Wan2.2 latent cache complete: "
            f"raw_written={written_raw}, raw_skipped={skipped_raw}, "
            f"motion_written={written_motion}, motion_skipped={skipped_motion}, "
            f"dst={dst_processed_dir}"
        )
    else:
        print(f"Wan2.2 latent cache complete: written={written}, skipped={skipped}, dst={dst_processed_dir}")


if __name__ == "__main__":
    main()
