import glob
import logging
import os
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # pragma: no cover - dependency is required at runtime
    load_safetensors_file = None

from .wan.wan_video_dit import WanModel, sinusoidal_embedding_1d


logger = logging.getLogger(__name__)


WAN22_TI2V_5B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


def _as_checkpoint_files(path_or_paths) -> Sequence[str]:
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, (list, tuple)):
        files = list(path_or_paths)
    else:
        path_or_paths = _resolve_path(path_or_paths)

    if isinstance(path_or_paths, (list, tuple)):
        files = list(path_or_paths)
    elif os.path.isdir(path_or_paths):
        files = sorted(glob.glob(os.path.join(path_or_paths, "*.safetensors")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_paths, "*.pt")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_paths, "*.pth")))
    else:
        files = [path_or_paths]
    return files


def _resolve_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    video2act_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_path = os.path.join(video2act_root, path)
    return local_path if os.path.exists(local_path) else path


def _load_state_dict(files: Iterable[str], map_location="cpu"):
    if load_safetensors_file is None:
        raise ImportError("safetensors is required to load Wan2.2 .safetensors checkpoints")

    state_dict = {}
    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        logger.info("Loading Wan checkpoint shard: %s", file_path)
        if file_path.endswith(".safetensors"):
            shard = load_safetensors_file(file_path, device=str(map_location))
        else:
            shard = torch.load(file_path, map_location=map_location)
            if isinstance(shard, dict) and "state_dict" in shard:
                shard = shard["state_dict"]
        state_dict.update(shard)
    return state_dict


class WanActionTransformer(nn.Module):
    """Wan2.2 TI2V DiT wrapper exposing token features for Video2Act conditioning."""

    def __init__(
        self,
        model: WanModel,
        feature_extraction_layers: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.feature_extraction_layers = feature_extraction_layers or len(model.blocks)
        self.dim = model.dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = hidden_states
        if x.ndim != 5:
            raise ValueError(f"WanActionTransformer expects [B, C, T, H, W] latents, got {tuple(x.shape)}")
        if x.shape[1] != self.model.in_dim:
            raise ValueError(
                f"Wan2.2 TI2V expects {self.model.in_dim} latent channels, got {x.shape[1]}. "
                "Please make sure the dataset provides Wan2.2 TI2V latents."
            )

        x = x.to(dtype=self.dtype)
        timestep = timestep.to(device=x.device, dtype=x.dtype).flatten()
        context = encoder_hidden_states.to(device=x.device, dtype=x.dtype)

        t = self.model.time_embedding(sinusoidal_embedding_1d(self.model.freq_dim, timestep).to(x.dtype))
        t_mod = self.model.time_projection(t).unflatten(1, (6, self.model.dim))
        context = self.model.text_embedding(context)

        x, (frames, height, width) = self.model.patchify(x)
        freqs = torch.cat(
            [
                self.model.freqs[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                self.model.freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                self.model.freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1).to(x.device)

        for block in self.model.blocks[: self.feature_extraction_layers]:
            x = block(x, context, t_mod, freqs)

        return x, encoder_hidden_states


def load_wan_action_transformer(
    device: str,
    dit_path,
    loading_device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    feature_extraction_layers: Optional[int] = None,
    defer_device_placement: bool = True,
    **kwargs,
):
    files = _as_checkpoint_files(dit_path)
    if not files:
        raise ValueError(f"Cannot find Wan checkpoint files from dit_path={dit_path!r}")

    model = WanModel(**WAN22_TI2V_5B_CONFIG)
    state_dict = _load_state_dict(files, map_location=loading_device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded Wan2.2 TI2V DiT, missing_keys=%d, unexpected_keys=%d", len(missing), len(unexpected))
    if missing:
        logger.debug("Wan missing keys: %s", missing[:20])
    if unexpected:
        logger.debug("Wan unexpected keys: %s", unexpected[:20])

    wrapper = WanActionTransformer(model, feature_extraction_layers=feature_extraction_layers, dtype=dtype)
    wrapper.requires_grad_(False)
    wrapper.eval()
    if not defer_device_placement:
        wrapper.to(device=device, dtype=dtype)
    else:
        wrapper.to(device=loading_device, dtype=dtype)
    return wrapper
