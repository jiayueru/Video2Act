from pathlib import Path

import torch

from .wan_video_vae import WanVideoVAE38, WanVideoVAEStateDictConverter


def _load_state_dict(path: Path, dtype=None):
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading Wan2.2 VAE safetensors requires `safetensors`."
            ) from exc
        state_dict = load_file(str(path), device="cpu")
    else:
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and len(state_dict) == 1:
            for key in ("state_dict", "module", "model_state"):
                if key in state_dict:
                    state_dict = state_dict[key]
                    break

    if dtype is not None:
        state_dict = {
            key: value.to(dtype) if isinstance(value, torch.Tensor) else value
            for key, value in state_dict.items()
        }
    return state_dict


def load_wan22_vae(path, device="cuda:0", dtype=torch.bfloat16):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Wan2.2 VAE path does not exist: {path}")

    vae = WanVideoVAE38()
    state_dict = WanVideoVAEStateDictConverter(_load_state_dict(path, dtype=dtype))
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load Wan2.2 VAE weights cleanly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    vae = vae.to(device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    return vae
