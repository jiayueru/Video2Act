#!/usr/bin/env python3
"""
Video2Act policy with WAN Dual-Stream Integration.

Two modes, controlled by `use_video_token_compressor`:
  - Direct projection (False): WAN tokens → LayerNorm → video_adaptor. Dual-stream
    builds a separate motion_adaptor; outputs concatenated along token dim.
  - Token compressor (True):   WAN tokens → LayerNorm → VideoTokenCompressor →
    shared video_adaptor (input=token_compressor_dim). Dual-stream uses two
    compressors (raw + motion); their outputs are concatenated then projected.

Text stream (both modes): T5 encoder states → lang_adaptor.

Token counts (VAE stride=16, WAN patch_size=(1,2,2)):
  raw    1 frame  @ [480,832]: 480//16//2 × 832//16//2 = 15×26 = 390 tokens
  motion 16 frames @ [224,224]: 224//16//2 × 224//16//2 × 16 = 7×7×16 = 784 tokens
  total: 1174  → video_compressed_tokens: 1174
"""

import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from .wan_feature_adapter import load_wan_action_transformer
from .wan_lora import apply_lora_to_linear_modules, is_lora_state_key, lora_parameters
from .video_token_compressor import VideoTokenCompressor


logger = logging.getLogger(__name__)

try:
    from .framepack.wan_generate_video import merge_lora_weights
    from .framepack.fpack_generate_video import convert_lora_for_framepack
    from .framepack.networks import lora_framepack
except ImportError:
    merge_lora_weights = None
    convert_lora_for_framepack = None
    lora_framepack = None
    logger.warning("LoRA modules not available")


class Video2ActWanPolicy(nn.Module, CompatiblePyTorchModelHubMixin):
    """
    Video2Act policy with WAN (Wan2.2 TI2V-5B) world-model feature extraction.

    When `use_video_token_compressor=False`, WAN output tokens are directly
    projected to Video2Act hidden size via video_adaptor (in_features =
    wan_hidden_size = 3072).

    When `use_video_token_compressor=True`, WAN tokens are first compressed by
    a VideoTokenCompressor (one per stream in dual-stream mode) and then
    projected by a shared video_adaptor (in_features = token_compressor_dim).
    """

    def __init__(self,
                 action_dim: int,
                 pred_horizon: int,
                 config: Dict[str, Any],
                 lang_token_dim: int,
                 img_token_dim: int,
                 state_token_dim: int,
                 video_encoder_token_dim: int,
                 max_lang_cond_len: int,
                 img_cond_len: int,
                 img_pos_embed_config: list,
                 lang_pos_embed_config: list,
                 video_pos_embed_config: list,
                 video_config: Dict[str, Any],
                 use_wan_blocks: bool = True,
                 dtype=torch.bfloat16,
                 feature_extraction_layers: int = 20,
                 action_denoise_layers: int = 40,
                 llama_vec_dim: int = 4096,
                 defer_wan_device_placement: bool = True,
                 **kwargs):
        super().__init__()

        self.defer_wan_device_placement = defer_wan_device_placement
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.state_token_dim = state_token_dim
        self.lang_token_dim = lang_token_dim
        self.img_token_dim = img_token_dim
        self.use_wan_blocks = use_wan_blocks
        self.dtype = dtype
        self.llama_vec_dim = llama_vec_dim

        self.video_config = video_config
        self.video_encoder_token_dim = video_encoder_token_dim
        self.feature_extraction_layers = feature_extraction_layers
        self.action_denoise_layers = action_denoise_layers

        # WAN transformer parameters
        self.wan_dit_path = video_config.get('dit_path')
        self.wan_device = video_config.get('device', 'cuda:0')
        self.wan_attn_mode = video_config.get('attn_mode', 'torch')
        self.wan_fp8_scaled = video_config.get('fp8_scaled', True)
        self.wan_num_single_layers = video_config.get('num_single_layers', 0)
        lora_cfg = video_config.get('train_lora', {}) or {}
        self.wan_lora_enabled = bool(lora_cfg.get('enabled', False))
        self.wan_lora_rank = int(lora_cfg.get('rank', 8))
        self.wan_lora_alpha = float(lora_cfg.get('alpha', self.wan_lora_rank))
        self.wan_lora_dropout = float(lora_cfg.get('dropout', 0.0))
        self.wan_lora_target_patterns = lora_cfg.get('target_patterns', [
            r"model\.blocks\.\d+\.self_attn\.(q|k|v|o)$",
            r"model\.blocks\.\d+\.cross_attn\.(q|k|v|o)$",
        ])
        self.wan_lora_layers = []

        # WAN hidden size (24 heads × 128 head_dim = 3072)
        self.wan_hidden_size = 3072
        self.latent_window_size = video_config.get('latent_window_size', 1)
        self.video_size = video_config.get('video_size', [480, 832])

        # Video2Act core configuration
        core_config = config['core']
        self.original_hidden_size = core_config.get('hidden_size', 2048)
        self.depth = core_config.get('depth', 28)
        self.num_heads = core_config.get('num_heads', 32)

        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.use_lang = config.get('use_lang', True)

        # Dual-stream config
        self.use_dual_stream_framepack = config.get('use_dual_stream_framepack', False)
        self.dual_stream_config = config.get('dual_stream_framepack', {})

        # Per-stream extraction config; read from dual_stream_framepack.*_stream.extraction
        # Actual filter params (freq_low/high etc.) live in framepack.motion_extraction
        me_cfg = video_config.get('motion_extraction', {})
        self.motion_extraction_cfg = me_cfg

        # Token compressor mode (read primary key; fall back to legacy alias)
        self.use_video_token_compressor = bool(
            config.get('use_video_token_compressor',
                       config.get('use_dual_token_compressor', True))
        )
        if self.use_video_token_compressor:
            self.token_compressor_dim = config.get(
                'video_compressed_dim',
                config.get('framepack_compressed_dim', 512)
            )
        else:
            self.token_compressor_dim = None

        self._init_adaptors(config)
        self._init_noise_scheduler(config)
        self._init_wan_transformer()

        # ---- Video2Act core model ----
        video_compressed_tokens = config.get('video_compressed_tokens', config.get('video_compressed_tokens', 1174))

        from models.action_core.model import Video2ActCore
        self.model = Video2ActCore(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=core_config['hidden_size'],
            depth=core_config['depth'],
            num_heads=core_config['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            video_pos_embed_config=video_pos_embed_config,
            video_compressed_tokens=video_compressed_tokens,
            dtype=dtype,
        )

        # Single LayerNorm before adaptor (numerical stability, no compression)
        self.video_layernorm = nn.LayerNorm(self.wan_hidden_size, dtype=dtype)

        # Dual-stream: separate layernorm for motion stream
        if self.use_dual_stream_framepack:
            raw_cfg = self.dual_stream_config.get('raw_stream', {})
            mot_cfg = self.dual_stream_config.get('motion_stream', {})
            self.raw_num_frames    = raw_cfg.get('num_frames', 1)
            self.motion_num_frames  = mot_cfg.get('num_frames', 16)
            self.motion_latent_num_frames = mot_cfg.get('latent_num_frames', self.motion_num_frames)
            self.raw_video_size    = tuple(raw_cfg.get('video_size', self.video_size))
            self.motion_video_size  = tuple(mot_cfg.get('video_size', [224, 224]))
            self.raw_extraction    = raw_cfg.get('extraction', None)   # e.g. 'sobel'
            self.mot_extraction    = mot_cfg.get('extraction', None)   # e.g. 'fft'
            self.raw_latent_source = raw_cfg.get('latent_source', 'raw')
            self.motion_layernorm  = nn.LayerNorm(self.wan_hidden_size, dtype=dtype)

            raw_tok   = self.raw_num_frames * (self.raw_video_size[0] // 16 // 2) * (self.raw_video_size[1] // 16 // 2)
            mot_tok   = self.motion_latent_num_frames * (self.motion_video_size[0] // 16 // 2) * (self.motion_video_size[1] // 16 // 2)
            logger.info(f"✅ Dual-Stream WAN inputs:")
            logger.info(f"   raw    {self.raw_num_frames}f @ {self.raw_video_size} → {raw_tok} tokens")
            logger.info(f"   motion {self.motion_num_frames} raw frames -> {self.motion_latent_num_frames} latent frames @ {self.motion_video_size} → {mot_tok} tokens")
            logger.info(f"   total  {raw_tok + mot_tok} WAN tokens before adaptor/compression")
        else:
            self.raw_num_frames = None
            self.motion_num_frames = None
            self.raw_video_size = None
            self.motion_video_size = None
            self.motion_layernorm = None
            T = self.latent_window_size
            tok = T * (self.video_size[0] // 16 // 2) * (self.video_size[1] // 16 // 2)
            logger.info(f"✅ Single-Stream WAN (no token compressor): {T}f → {tok} tokens")

        if self.use_video_token_compressor:
            self._init_token_compressors(config)
        else:
            self.video_token_compressor = None
            self.motion_token_compressor = None

        # Eval cache
        self.cache_enabled = False
        self.cache_ratio = 2
        self.cache_count = 0
        self.cached_adapted_hidden_states = None

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_adaptors(self, config):
        core_hidden = config['core']['hidden_size']
        video_adaptor_type = config.get('video_adaptor', config.get('framepack_adaptor'))

        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], self.lang_token_dim, core_hidden)
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], self.img_token_dim, core_hidden)
        self.state_adaptor = self.build_condition_adapter(
            config.get('state_adaptor', 'mlp2x_gelu'), 128 * 2, core_hidden)

        if self.use_video_token_compressor:
            # Compressor reduces wan_hidden_size → token_compressor_dim before adaptor.
            # Dual-stream concatenates compressor outputs and shares a single adaptor.
            self.video_adaptor = self.build_condition_adapter(
                video_adaptor_type, self.token_compressor_dim, core_hidden)
            self.motion_adaptor = None
        else:
            # Direct projection: wan_hidden_size → core_hidden.
            self.video_adaptor = self.build_condition_adapter(
                video_adaptor_type, self.wan_hidden_size, core_hidden)
            if self.use_dual_stream_framepack:
                self.motion_adaptor = self.build_condition_adapter(
                    video_adaptor_type, self.wan_hidden_size, core_hidden)
            else:
                self.motion_adaptor = None

    def _init_token_compressors(self, config):
        """Build VideoTokenCompressor(s). Dual-stream uses one per stream."""
        depth = config.get('token_compressor_depth', 3)
        heads = config.get('token_compressor_heads', 8)
        default_num_latents = config.get(
            'video_compressed_tokens',
            config.get('framepack_compressed_tokens', 448)
        )

        if self.use_dual_stream_framepack:
            raw_cfg = self.dual_stream_config.get('raw_stream', {})
            motion_cfg = self.dual_stream_config.get('motion_stream', {})

            self.raw_compressed_tokens = raw_cfg.get('compressed_tokens', 224)
            self.motion_compressed_tokens = motion_cfg.get('compressed_tokens', 224)
            if self.motion_compressed_tokens <= 0:
                raise ValueError(
                    "motion_stream.compressed_tokens must be positive; check raw/motion token split."
                )

            raw_compressed_dim = raw_cfg.get('compressed_dim', self.token_compressor_dim)
            motion_compressed_dim = motion_cfg.get('compressed_dim', self.token_compressor_dim)
            if raw_compressed_dim != motion_compressed_dim:
                raise ValueError(
                    "WAN dual token compressor currently requires raw_stream.compressed_dim "
                    "and motion_stream.compressed_dim to match."
                )
            if raw_compressed_dim != self.token_compressor_dim:
                raise ValueError(
                    "WAN dual token compressor requires stream compressed_dim to equal "
                    "model.video_compressed_dim."
                )

            self.video_token_compressor = VideoTokenCompressor(
                dim=raw_compressed_dim,
                depth=raw_cfg.get('compressor_depth', depth),
                condition_dim=self.wan_hidden_size,
                dim_head=64,
                heads=raw_cfg.get('compressor_heads', heads),
                num_latents=self.raw_compressed_tokens,
                num_frame=raw_cfg.get('compressor_num_frames', self.raw_num_frames or 1),
                num_time_embeds=raw_cfg.get('compressor_num_time_embeds', self.raw_num_frames or 1),
                use_temporal=raw_cfg.get('use_temporal_attention', False),
                dtype=self.dtype,
            )
            self.motion_token_compressor = VideoTokenCompressor(
                dim=motion_compressed_dim,
                depth=motion_cfg.get('compressor_depth', depth),
                condition_dim=self.wan_hidden_size,
                dim_head=64,
                heads=motion_cfg.get('compressor_heads', heads),
                num_latents=self.motion_compressed_tokens,
                num_frame=motion_cfg.get('compressor_num_frames', self.motion_latent_num_frames or 1),
                num_time_embeds=motion_cfg.get(
                    'compressor_num_time_embeds', self.motion_latent_num_frames or 1),
                use_temporal=motion_cfg.get('use_temporal_attention', True),
                dtype=self.dtype,
            )
            logger.info(
                "✅ WAN dual token compressor enabled: raw %s tokens + motion %s tokens = %s tokens",
                self.raw_compressed_tokens,
                self.motion_compressed_tokens,
                self.raw_compressed_tokens + self.motion_compressed_tokens,
            )
        else:
            self.video_token_compressor = VideoTokenCompressor(
                dim=self.token_compressor_dim,
                depth=depth,
                condition_dim=self.wan_hidden_size,
                dim_head=64,
                heads=heads,
                num_latents=default_num_latents,
                num_frame=config.get('token_compressor_num_frames', 1),
                num_time_embeds=config.get('token_compressor_num_time_embeds', 1),
                use_temporal=config.get('use_temporal_attention', False),
                dtype=self.dtype,
            )
            self.motion_token_compressor = None
            logger.info(
                "✅ WAN single token compressor enabled: WAN tokens 3072d -> %s tokens x %sd -> Video2Act",
                default_num_latents,
                self.token_compressor_dim,
            )

    def _init_noise_scheduler(self, config):
        ns = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=ns['num_train_timesteps'],
            beta_schedule=ns['beta_schedule'],
            prediction_type=ns['prediction_type'],
            clip_sample=ns['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=ns['num_train_timesteps'],
            beta_schedule=ns['beta_schedule'],
            prediction_type=ns['prediction_type'],
        )
        self.num_train_timesteps = ns['num_train_timesteps']
        self.num_inference_timesteps = ns['num_inference_timesteps']
        self.prediction_type = ns['prediction_type']

    def _init_wan_transformer(self):
        """Load frozen Wan2.2 TI2V-5B as world-model feature extractor."""
        if self.wan_dit_path is not None:
            logger.info(f"Loading Wan2.2 TI2V DiT from: {self.wan_dit_path}")
            if self.defer_wan_device_placement:
                logger.info("Deferring WAN device placement — accelerator.prepare handles it later")
            loading_device = "cpu" if self.defer_wan_device_placement else self.wan_device
            self.wan_transformer = load_wan_action_transformer(
                device=self.wan_device,
                dit_path=self.wan_dit_path,
                loading_device=loading_device,
                dtype=self.dtype,
                feature_extraction_layers=self.feature_extraction_layers,
                defer_device_placement=self.defer_wan_device_placement,
            )
            self.wan_transformer.requires_grad_(False)
            if self.wan_lora_enabled:
                self.wan_lora_layers = apply_lora_to_linear_modules(
                    self.wan_transformer,
                    target_patterns=self.wan_lora_target_patterns,
                    rank=self.wan_lora_rank,
                    alpha=self.wan_lora_alpha,
                    dropout=self.wan_lora_dropout,
                )
                if not self.wan_lora_layers:
                    raise ValueError("WAN LoRA enabled but no Linear layers matched target_patterns")
                logger.info(
                    "✅ WAN LoRA trainable: rank=%s alpha=%s dropout=%s layers=%s",
                    self.wan_lora_rank, self.wan_lora_alpha, self.wan_lora_dropout, len(self.wan_lora_layers)
                )
            if hasattr(self.wan_transformer, 'model') and hasattr(self.wan_transformer.model, 'blocks'):
                logger.info(f"   WAN DiT blocks: {len(self.wan_transformer.model.blocks)} "
                            f"(using first {self.feature_extraction_layers})")
        else:
            self.wan_transformer = None
            logger.info("No WAN DIT path provided")

    def build_condition_adapter(self, projector_type: str, in_features: int, out_features: int):
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            m = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if m:
                depth = int(m.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, depth):
                    modules += [nn.GELU(approximate="tanh"), nn.Linear(out_features, out_features)]
                projector = nn.Sequential(*modules)
            else:
                raise ValueError(f'Unknown projector type: {projector_type}')
        for mod in (projector if isinstance(projector, nn.Sequential) else [projector]):
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
        return projector

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def enable_cache(self, cache_ratio: int = 2):
        self.cache_enabled = True
        self.cache_ratio = cache_ratio
        self.cache_count = 0

    def disable_cache(self):
        self.cache_enabled = False
        self.cached_adapted_hidden_states = None

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _wan_forward(self, hidden_states, dummy_timesteps, encoder_hidden_states,
                     encoder_attention_mask, device):
        """Single WAN forward. Returns [B, N_tokens, wan_hidden_size]."""
        out, _ = self.wan_transformer(
            hidden_states=hidden_states,
            timestep=dummy_timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return out  # [B, N_tokens, 3072]

    def _get_wan_text_context(self, batch_data, batch_size, device, dtype):
        """Return WAN text context, or an empty context when eval disables it."""
        if batch_data is None:
            batch_data = {}
        enc_hs = batch_data.get("llama_vec")
        enc_mask = batch_data.get("llama_attention_mask")
        if enc_hs is None:
            enc_hs = torch.empty(
                batch_size, 0, self.llama_vec_dim, device=device, dtype=dtype
            )
            enc_mask = torch.empty(batch_size, 0, device=device, dtype=dtype)
            return enc_hs, enc_mask
        enc_hs = enc_hs.squeeze(1).to(device=device, dtype=dtype)
        if enc_mask is None:
            enc_mask = torch.ones(
                enc_hs.shape[:2], device=device, dtype=dtype
            )
        else:
            enc_mask = enc_mask.squeeze(1).to(device=device, dtype=dtype)
        return enc_hs, enc_mask

    def _resize_latents(self, latents, target_h, target_w):
        """Bilinear resize [B, C, T, H, W] latents to target spatial dims."""
        B, C, T, H, W = latents.shape
        if H == target_h and W == target_w:
            return latents
        x = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)

    def _apply_temporal_fft(self, latents: torch.Tensor) -> torch.Tensor:
        """Temporal FFT bandpass / lowpass filter along the T dimension.

        Keeps frequencies in [freq_low, freq_high] Hz (bandpass) or
        [0, freq_high] Hz (lowpass). DC (static background) is removed
        in bandpass mode, which leaves only the motion signal.

        Args:
            latents: [B, C, T, H, W]
        Returns:
            filtered latents, same shape and dtype
        """
        me = self.motion_extraction_cfg
        filter_type = me.get('filter_type', 'bandpass')
        freq_low    = me.get('freq_low',  1.0)
        freq_high   = me.get('freq_high', 10.0)
        fps         = self.video_config.get('fps', 16)

        T = latents.shape[2]
        orig_dtype = latents.dtype
        x = latents.float()

        fft   = torch.fft.rfft(x, dim=2)                              # [B, C, T//2+1, H, W]
        freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(latents.device) # [T//2+1]

        if filter_type == 'bandpass':
            mask = (freqs >= freq_low) & (freqs <= freq_high)
        else:  # lowpass
            mask = freqs <= freq_high

        fft      = fft * mask[None, None, :, None, None].float()
        filtered = torch.fft.irfft(fft, n=T, dim=2)
        return filtered.to(orig_dtype)

    def _apply_sobel(self, latents: torch.Tensor) -> torch.Tensor:
        """Spatial Sobel edge detection applied per frame across all latent channels.

        Outputs the gradient magnitude so that the channel count and spatial
        dimensions are unchanged — WAN still receives [B, 48, T, H, W].

        Args:
            latents: [B, C, T, H, W]
        Returns:
            edge-magnitude latents, same shape and dtype
        """
        B, C, T, H, W = latents.shape
        orig_dtype = latents.dtype
        device = latents.device

        x = latents.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3).expand(C, 1, 3, 3)

        gx  = F.conv2d(x, kx, padding=1, groups=C)
        gy  = F.conv2d(x, ky, padding=1, groups=C)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        return mag.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).to(orig_dtype)

    def _apply_scharr(self, latents: torch.Tensor) -> torch.Tensor:
        """Spatial Scharr edge detection applied per frame across all latent channels.

        Scharr uses stronger center weights than Sobel, improving rotational symmetry
        while preserving [B, C, T, H, W] for the downstream WAN transformer.
        """
        B, C, T, H, W = latents.shape
        orig_dtype = latents.dtype
        device = latents.device

        x = latents.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        kx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                          device=device).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        ky = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]],
                          device=device).view(1, 1, 3, 3).expand(C, 1, 3, 3)

        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        return mag.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).to(orig_dtype)

    def _apply_laplace(self, latents: torch.Tensor) -> torch.Tensor:
        """Spatial Laplacian edge detection applied per frame across all latent channels.

        Outputs absolute second-derivative response while preserving [B, C, T, H, W].
        """
        B, C, T, H, W = latents.shape
        orig_dtype = latents.dtype
        device = latents.device

        x = latents.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=device).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        edge = F.conv2d(x, k, padding=1, groups=C).abs()
        return edge.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).to(orig_dtype)

    def _apply_moft(self, latents: torch.Tensor) -> torch.Tensor:
        """MOFT: zero out channels with low temporal variance, keep top-k motion channels.

        prop_motion controls the fraction of channels kept (e.g. 0.04 → top 2/48).

        Args:
            latents: [B, C, T, H, W]
        Returns:
            masked latents, same shape and dtype
        """
        prop = self.motion_extraction_cfg.get('prop_motion', 0.04)
        var  = latents.var(dim=2)                    # [B, C, H, W]
        mean_var = var.mean(dim=(0, 2, 3))            # [C]
        k    = max(1, int(latents.shape[1] * prop))
        top_idx = mean_var.topk(k).indices
        mask = torch.zeros(latents.shape[1], device=latents.device, dtype=latents.dtype)
        mask[top_idx] = 1.0
        return latents * mask[None, :, None, None, None]

    def _apply_stream_extraction(self, latents: torch.Tensor, method: str) -> torch.Tensor:
        """Apply the named extraction to latents for one stream.

        Supported methods:
          'fft'       — temporal bandpass/lowpass filter (remove static background)
          'sobel'     — spatial Sobel edge magnitude (enhance spatial motion edges)
          'scharr'    — spatial Scharr edge magnitude (Sobel-like, stronger weights)
          'laplace'   — spatial Laplacian edge magnitude (second-derivative edges)
          'fft+sobel' — FFT then Sobel
          'moft'      — keep top-k temporally-variant channels

        Args:
            latents: [B, C, T, H, W]
            method:  extraction method string, or None/'' to skip
        Returns:
            processed latents, same shape and dtype
        """
        if not method:
            return latents
        if method == 'fft':
            return self._apply_temporal_fft(latents)
        elif method == 'sobel':
            return self._apply_sobel(latents)
        elif method == 'scharr':
            return self._apply_scharr(latents)
        elif method == 'laplace':
            return self._apply_laplace(latents)
        elif method in ('fft+sobel', 'sobel+fft'):
            return self._apply_sobel(self._apply_temporal_fft(latents))
        elif method == 'moft':
            return self._apply_moft(latents)
        else:
            logger.warning("Unknown stream extraction method '%s', skipping", method)
            return latents

    def _select_motion_latent_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply temporal ablation to motion latents after loading cached chunks."""
        mot_cfg = self.dual_stream_config.get("motion_stream", {})
        target_t = int(mot_cfg.get("latent_num_frames", latents.shape[2]))
        mode = mot_cfg.get("latent_frame_select", "last")
        current_t = latents.shape[2]
        if current_t == target_t:
            return latents
        if current_t < target_t:
            raise ValueError(
                f"image_latents_motion has only {current_t} frames after cache concat, "
                f"but latent_num_frames={target_t}. Increase cache_concat_chunks or lower T_l."
            )
        if mode == "uniform":
            idx = torch.linspace(0, current_t - 1, target_t, device=latents.device).round().long()
            return latents.index_select(2, idx)
        if mode == "first":
            return latents[:, :, :target_t]
        if mode == "center":
            start = max(0, (current_t - target_t) // 2)
            return latents[:, :, start:start + target_t]
        if mode != "last":
            logger.warning("Unknown latent_frame_select=%s, falling back to last", mode)
        return latents[:, :, -target_t:]

    @staticmethod
    def _require_latent_shape(name: str, latents: torch.Tensor,
                              expected_t: int, expected_h: int, expected_w: int):
        if latents.ndim != 5:
            raise ValueError(f"{name} must be [B, C, T, H, W], got {tuple(latents.shape)}")
        _, _, t, h, w = latents.shape
        if (t, h, w) != (expected_t, expected_h, expected_w):
            raise ValueError(
                f"{name} has wrong latent shape {tuple(latents.shape)}; "
                f"expected temporal/spatial {(expected_t, expected_h, expected_w)}."
            )

    # ------------------------------------------------------------------
    # adapt_conditions — dual or single stream, no token compressor
    # ------------------------------------------------------------------

    def adapt_conditions(self, img_tokens, batch_data=None):
        """
        Run WAN feature extraction and project to Video2Act hidden size.

        Direct projection (use_video_token_compressor=False):
          dual-stream:    raw → ln → video_adaptor; motion → ln → motion_adaptor;
                          cat → [B, N_raw+N_mot, core_hidden]
          single-stream:  [B, N, 3072] → ln → video_adaptor → [B, N, core_hidden]

        Token compressor (use_video_token_compressor=True):
          dual-stream:    raw → ln → video_token_compressor; motion → ln →
                          motion_token_compressor; cat → shared video_adaptor
          single-stream:  [B, N, 3072] → ln → video_token_compressor → video_adaptor

        Text policy:
          batch_data["llama_vec"] is used only as WAN DiT text context.
          Video2Act blocks receive T5 language embeddings through lang_tokens/lang_adaptor.
        """
        B = img_tokens.shape[0]
        device = img_tokens.device
        dtype = self.dtype

        # ---- Eval cache ----
        if self.cache_enabled and not self.training:
            self.cache_count += 1
            if (self.cache_count % self.cache_ratio != 1) and self.cached_adapted_hidden_states is not None:
                return self.cached_adapted_hidden_states

        # ---- Shared WAN inputs ----
        dummy_ts = torch.zeros((B,), device=device, dtype=torch.long)
        enc_hs, enc_mask = self._get_wan_text_context(batch_data, B, device, dtype)

        # ---- Feature extraction ----
        if self.use_dual_stream_framepack and self.motion_layernorm is not None:
            # Raw stream: normally image_latents_head encodes 1 frame at
            # [480,832] → [B,C,1,30,52]. For raw-resolution ablations, reuse
            # the current motion latent frame as [B,C,1,14,14] without another
            # precache pass.
            if self.raw_latent_source == "motion_current":
                if "image_latents_motion" not in batch_data:
                    raise KeyError(
                        "raw_stream.latent_source=motion_current requires "
                        "batch_data['image_latents_motion']."
                    )
                raw_hs = batch_data["image_latents_motion"].to(device=device, dtype=dtype)
                if raw_hs.ndim == 6:
                    raw_hs = raw_hs.squeeze(1)
                raw_hs = self._select_motion_latent_frames(raw_hs)[:, :, -self.raw_num_frames:]
            else:
                raw_hs = batch_data["image_latents_head"].to(device=device, dtype=dtype)
            if raw_hs.ndim == 6:
                raw_hs = raw_hs.squeeze(1)
            self._require_latent_shape(
                "image_latents_head",
                raw_hs,
                expected_t=self.raw_num_frames,
                expected_h=self.raw_video_size[0] // 16,
                expected_w=self.raw_video_size[1] // 16,
            )
            raw_hs = self._apply_stream_extraction(raw_hs, self.raw_extraction)
            raw_out = self._wan_forward(raw_hs, dummy_ts, enc_hs, enc_mask, device)
            raw_norm = self.video_layernorm(raw_out)

            # Motion stream: image_latents_motion encodes the 61-frame window
            # ending at the current step via VAE /4 temporal compression:
            # [B, 1, C, 16, 14, 14] → [B, C, 16, 14, 14].
            if "image_latents_motion" not in batch_data:
                raise KeyError(
                    "Dual-stream WAN requires batch_data['image_latents_motion']. "
                    "Run/write motion latent precache before training."
                )
            mot_hs = batch_data["image_latents_motion"].to(device=device, dtype=dtype)
            if mot_hs.ndim == 6:
                mot_hs = mot_hs.squeeze(1)
            mot_hs = self._select_motion_latent_frames(mot_hs)
            expected_motion_latent_frames = self.dual_stream_config.get(
                'motion_stream', {}
            ).get('latent_num_frames', 16)
            self._require_latent_shape(
                "image_latents_motion",
                mot_hs,
                expected_t=expected_motion_latent_frames,
                expected_h=self.motion_video_size[0] // 16,
                expected_w=self.motion_video_size[1] // 16,
            )
            mot_hs = self._apply_stream_extraction(mot_hs, self.mot_extraction)
            mot_out = self._wan_forward(mot_hs, dummy_ts, enc_hs, enc_mask, device)
            mot_norm = self.motion_layernorm(mot_out)

            if self.use_video_token_compressor:
                raw_compressed = self.video_token_compressor(raw_norm)
                motion_compressed = self.motion_token_compressor(mot_norm)
                adapted_hidden_states = self.video_adaptor(
                    torch.cat([raw_compressed, motion_compressed], dim=1)
                )
            else:
                raw_tokens = self.video_adaptor(raw_norm)
                mot_tokens = self.motion_adaptor(mot_norm)
                adapted_hidden_states = torch.cat([raw_tokens, mot_tokens], dim=1)
        else:
            # Single stream
            image_latents = batch_data["image_latents_head"].to(device=device, dtype=dtype)
            hidden_states = image_latents.squeeze(1)
            if hidden_states.ndim == 6:
                hidden_states = hidden_states.squeeze(1)
            target_h = max(1, self.video_size[0] // 16)
            target_w = max(1, self.video_size[1] // 16)
            hidden_states = self._resize_latents(hidden_states, target_h, target_w)
            out = self._wan_forward(hidden_states, dummy_ts, enc_hs, enc_mask, device)
            norm = self.video_layernorm(out)
            if self.use_video_token_compressor:
                adapted_hidden_states = self.video_adaptor(self.video_token_compressor(norm))
            else:
                adapted_hidden_states = self.video_adaptor(norm)

        if self.cache_enabled and not self.training:
            self.cached_adapted_hidden_states = adapted_hidden_states

        return adapted_hidden_states

    # ------------------------------------------------------------------
    # Training & sampling
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss_realtime(*args, **kwargs)

    def compute_loss_realtime(self, lang_tokens, lang_attn_mask, img_tokens,
                              state_tokens, action_gt, ctrl_freqs,
                              batch_data=None, action_mask=None, **kwargs):
        batch_size = img_tokens.shape[0]
        device = img_tokens.device

        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        adapted_hs = self.adapt_conditions(img_tokens, batch_data=batch_data)

        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        state_action_traj = self.state_adaptor(state_action_traj)

        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img  = self.img_adaptor(img_tokens)

        pred = self.model(state_action_traj, ctrl_freqs, timesteps,
                          lang_c=adapted_lang, img_c=adapted_img,
                          video_c=adapted_hs,
                          lang_mask=lang_attn_mask)

        target = noise if self.prediction_type == 'epsilon' else action_gt
        return F.mse_loss(pred, target)

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs, images=None, instruction_text=None,
                       batch_data=None, **kwargs):
        with torch.no_grad():
            adapted_hs = self.adapt_conditions(img_tokens, batch_data=batch_data)
            adapted_lang = self.lang_adaptor(lang_tokens)
            adapted_img  = self.img_adaptor(img_tokens)
            state_traj   = self.state_adaptor(torch.cat([state_tokens, action_mask], dim=2))
            action_pred  = self.conditional_sample_wan(
                lang_cond=adapted_lang, lang_attn_mask=lang_attn_mask,
                img_cond=adapted_img, motion_cond=adapted_hs,
                state_traj=state_traj, action_mask=action_mask, ctrl_freqs=ctrl_freqs,
            )
        return action_pred

    def conditional_sample_wan(self, lang_cond, lang_attn_mask, img_cond, motion_cond,
                               state_traj, action_mask, ctrl_freqs):
        device = state_traj.device
        dtype  = state_traj.dtype
        noisy_action = torch.randn(
            (state_traj.shape[0], self.pred_horizon, self.action_dim), dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        for t in self.noise_scheduler_sample.timesteps:
            action_traj = self.state_adaptor(torch.cat([noisy_action, action_mask], dim=2))
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            model_output = self.model(
                state_action_traj, ctrl_freqs, t.unsqueeze(-1).to(device),
                lang_c=lang_cond, img_c=img_cond, video_c=motion_cond,
                lang_mask=lang_attn_mask)
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(dtype)

        return noisy_action * action_mask

    # ------------------------------------------------------------------
    # Parameter / checkpoint helpers
    # ------------------------------------------------------------------

    def get_trainable_parameters(self):
        components = {
            'action_core':     self.model.parameters(),
            'lang_adaptor':    self.lang_adaptor.parameters(),
            'img_adaptor':     self.img_adaptor.parameters(),
            'video_adaptor':   self.video_adaptor.parameters(),
            'video_layernorm': self.video_layernorm.parameters(),
            'state_adaptor':   self.state_adaptor.parameters(),
        }
        if self.use_dual_stream_framepack and self.motion_layernorm is not None:
            components['motion_layernorm'] = self.motion_layernorm.parameters()
        if self.motion_adaptor is not None:
            components['motion_adaptor'] = self.motion_adaptor.parameters()
        if self.video_token_compressor is not None:
            components['video_token_compressor'] = self.video_token_compressor.parameters()
        if self.motion_token_compressor is not None:
            components['motion_token_compressor'] = self.motion_token_compressor.parameters()
        if self.wan_lora_enabled and self.wan_transformer is not None:
            components['wan_lora'] = lora_parameters(self.wan_transformer)

        all_params = []
        for name, params in components.items():
            p = [x for x in params if x.requires_grad]
            all_params.extend(p)
            logger.debug(f"  {name}: {len(p)} trainable tensors")
        logger.debug(f"Total trainable: {sum(x.numel() for x in all_params):,}")
        return all_params

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Exclude frozen wan_transformer from checkpoints."""
        full = super().state_dict(destination, prefix, keep_vars)
        filtered = {
            k: v for k, v in full.items()
            if not k.startswith('wan_transformer' + '.') or is_lora_state_key(k)
        }
        excluded = len(full) - len(filtered)
        if excluded:
            logger.info(f"Excluded {excluded} wan_transformer tensors from state_dict")
        return filtered

    def merge_lora(self, lora_weights=None, lora_multiplier=1.0, save_merged_model=None,
                   include_patterns=None, exclude_patterns=None, lycoris=False):
        if lora_weights is None and hasattr(self, 'video_config'):
            cfg = self.video_config or {}
            lora_weights      = cfg.get('lora_weight')
            lora_multiplier   = cfg.get('lora_multiplier', lora_multiplier)
            include_patterns  = cfg.get('include_patterns', include_patterns)
            exclude_patterns  = cfg.get('exclude_patterns', exclude_patterns)
            lycoris           = cfg.get('lycoris', lycoris)
            save_merged_model = cfg.get('save_merged_model', save_merged_model)

        if not merge_lora_weights or not self.wan_transformer or not lora_weights:
            return
        import argparse
        args = argparse.Namespace(
            lora_weight=[lora_weights] if isinstance(lora_weights, str) else lora_weights,
            lora_multiplier=[lora_multiplier] if isinstance(lora_multiplier, (int, float)) else lora_multiplier,
            save_merged_model=save_merged_model,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            lycoris=lycoris,
        )
        try:
            merge_lora_weights(lora_framepack, self.wan_transformer, args,
                               torch.device(self.wan_device), convert_lora_for_framepack)
            logger.info("✅ LoRA merged")
        except Exception as e:
            logger.error(f"LoRA merge failed: {e}")
            raise
