"""
Lightweight encoder for WAN inference:
  - WAN VAE for encoding images/video to latents
  - WAN UMT5 text encoder for encoding instructions to llama_vec
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add Lightewm to path
_LIGHTEWM_ROOT = str(Path(__file__).resolve().parents[4] / "Lightewm")
if _LIGHTEWM_ROOT not in sys.path:
    sys.path.insert(0, _LIGHTEWM_ROOT)

from lightewm.model.wan.wan_video_vae import WanVideoVAE, WanVideoVAE38, WanVideoVAEStateDictConverter
from lightewm.model.wan.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer

MAX_SEQ_LEN = 512


class FramepackLightweightEncoder:
    def __init__(
        self,
        vae_path=None,
        text_encoder1_path=None,
        text_encoder2_path=None,  # unused for WAN 2.2
        device="cuda",
        dtype=torch.bfloat16,
        load_vae=True,
        load_text_encoders=True,
        tokenizer_path=None,
        custom_system_prompt=None,
        guidance_scale=1.0,
        vae_chunk_size=None,
        vae_spatial_tile_sample_min_size=None,
    ):
        self.device = device
        self.dtype = dtype

        # Load VAE
        self.vae = None
        if load_vae and vae_path is not None:
            self.vae = self._load_vae(vae_path)

        # Load text encoder (UMT5)
        self.text_encoder1 = None
        self.text_encoder2 = None  # WAN 2.2 doesn't use a second text encoder
        self.tokenizer = None
        if load_text_encoders and text_encoder1_path is not None:
            self.text_encoder1, self.tokenizer = self._load_text_encoder(
                text_encoder1_path, tokenizer_path
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_vae(self, vae_path):
        state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = WanVideoVAEStateDictConverter(state_dict)
        # Wan2.2-TI2V-5B uses WanVideoVAE38 (encoder dim=160, decoder dec_dim=256, z_dim=48)
        vae = WanVideoVAE38()
        vae.load_state_dict(state_dict)
        # Move VAE to the same device as other encoders.
        # self.mean/std/scale are plain tensors but WanVideoVAE.encode() already handles
        # them via scale[i].to(device=mu.device), so GPU is safe.
        vae = vae.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        return vae

    def _load_text_encoder(self, encoder_path, tokenizer_path=None):
        state_dict = torch.load(encoder_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        encoder = WanTextEncoder()
        encoder.load_state_dict(state_dict)
        encoder = encoder.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)

        if tokenizer_path is None:
            # Try a default path next to the encoder file
            tokenizer_path = str(Path(encoder_path).parent / "google" / "umt5-xxl")
        try:
            tokenizer = HuggingfaceTokenizer(
                name=tokenizer_path, seq_len=MAX_SEQ_LEN, clean="whitespace", extra_ids=0
            )
        except Exception:
            tokenizer = None

        return encoder, tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_text(self, text_instruction):
        """Returns (positive_ctx, None) where positive_ctx contains llama_vec etc."""
        if self.text_encoder1 is None or self.tokenizer is None:
            raise RuntimeError("Text encoder not initialized")

        ids, mask = self.tokenizer(text_instruction, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            emb = self.text_encoder1(ids, mask)  # [1, L, 4096]

        # Zero out padding
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            emb[i, v:] = 0

        # Crop or pad to MAX_SEQ_LEN
        L = emb.shape[1]
        if L >= MAX_SEQ_LEN:
            emb = emb[:, :MAX_SEQ_LEN, :]
            mask = mask[:, :MAX_SEQ_LEN]
        else:
            pad = MAX_SEQ_LEN - L
            emb = torch.cat([emb, emb.new_zeros(emb.shape[0], pad, emb.shape[2])], dim=1)
            mask = torch.cat([mask, mask.new_zeros(mask.shape[0], pad)], dim=1)

        ctx = {
            "llama_vec": emb.cpu(),
            "llama_attention_mask": mask.cpu().float(),
            "clip_l_pooler": torch.zeros(1, 1, 1, dtype=self.dtype),
        }
        return ctx, None

    def encode_image_with_vae(self, tensor):
        """
        Encode image/video tensor using WAN VAE.
        Input:  [1, 3, T, H, W]  float32 in [-1, 1]
        Output: [1, 16, T', H', W']  on CPU
        VAE always runs on CPU because WanVideoVAE's scale tensors (mean/std) are
        plain torch tensors that don't move with .to(device).
        """
        if self.vae is None:
            raise RuntimeError("VAE not initialized")

        # WanVideoVAE.encode expects a list of [3, T, H, W] tensors
        videos = [tensor[0].float().cpu()]  # encode() internally moves to self.device
        with torch.no_grad():
            latents = self.vae.encode(videos, device=self.device)  # [1, 16, T', H', W']
        return latents.cpu()

    def resize_image_to_bucket(self, image_rgb, target_wh):
        """Resize numpy HxWx3 image to (width, height) using area interpolation."""
        w, h = target_wh
        resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
        return resized  # HxWx3 numpy
