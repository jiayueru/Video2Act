import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder

try:
    from models.framepack_lightweight_encoder import (
        FramepackLightweightEncoder as Video2ActLightweightEncoder,
    )
except ImportError:
    Video2ActLightweightEncoder = None

_WAN_DIT_FALLBACK = os.environ.get("VIDEO2ACT_WAN_DIT_PATH", "")
_WAN_VAE_FALLBACK = os.environ.get("VIDEO2ACT_WAN_VAE_PATH", "")
_LEGACY_WAN_DIT_NAMES = (
    "Wan2.2-5B-Robot/checkpoint.safetensors",
)
_LEGACY_WAN_VAE_NAMES = (
    "wan-t2v-5b/Wan2.2_VAE.pth",
)


def _remap_legacy_wan_path(path, legacy_patterns, fallback_path, field_name):
    if not path:
        return path

    path_str = str(path)
    if os.path.exists(path_str):
        return path

    is_legacy = any(pat in path_str for pat in legacy_patterns)
    if not is_legacy:
        return path

    if fallback_path and os.path.exists(fallback_path):
        print(f"🔁 Remap missing {field_name}: {path_str} -> {fallback_path}")
        return fallback_path

    print(
        f"⚠ Missing {field_name}: {path_str}; set VIDEO2ACT_WAN_DIT_PATH/VIDEO2ACT_WAN_VAE_PATH if needed."
    )
    return path


def _sanitize_runtime_wan_paths(args):
    if not isinstance(args, dict):
        return

    video_cfg = args.get("framepack")
    if not isinstance(video_cfg, dict):
        return

    video_cfg["dit_path"] = _remap_legacy_wan_path(
        video_cfg.get("dit_path"),
        _LEGACY_WAN_DIT_NAMES,
        _WAN_DIT_FALLBACK,
        "framepack.dit_path",
    )
    video_cfg["vae_path"] = _remap_legacy_wan_path(
        video_cfg.get("vae_path"),
        _LEGACY_WAN_VAE_NAMES,
        _WAN_VAE_FALLBACK,
        "framepack.vae_path",
    )


def load_pretrained_weights_simple(model, pretrained_path, logger=None, checkpoint_type="deepspeed"):
    """Non-strict weight loader.

    Args:
        model: model
        pretrained_path: checkpoint path (file or directory)
        logger: logger
        checkpoint_type: one of "deepspeed", "final", "ema"

    Returns:
        bool: success
    """
    log_fn = logger.info if logger and hasattr(logger, 'info') else print

    if checkpoint_type is None:
        checkpoint_type = "deepspeed"

    try:
        log_fn(f"Loading from: {pretrained_path} (type: {checkpoint_type})")

        # Step 1: pick the file to load
        if os.path.isfile(pretrained_path):
            model_path = pretrained_path
        elif os.path.isdir(pretrained_path):
            type_map = {
                "deepspeed": "pytorch_model/mp_rank_00_model_states.pt",
                "final": "pytorch_model.bin",
                "ema": "ema/pytorch_model.bin"
            }
            if checkpoint_type.lower() not in type_map:
                raise ValueError(f"Invalid type: {checkpoint_type}. Use: deepspeed/final/ema")

            model_path = os.path.join(pretrained_path, type_map[checkpoint_type.lower()])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"File not found: {model_path}")
        else:
            raise FileNotFoundError(f"Path not found: {pretrained_path}")

        # Step 2: load checkpoint
        log_fn(f"Loading: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Step 3: extract state_dict (multiple formats)
        if isinstance(checkpoint, dict):
            if 'module' in checkpoint:
                state_dict = checkpoint['module']  # DeepSpeed
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']  # standard
            else:
                state_dict = checkpoint  # raw state_dict
        else:
            state_dict = checkpoint

        # Step 4: move to model device and load
        device = next(model.parameters()).device
        state_dict = {k: v.to(device) if hasattr(v, 'to') else v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Step 5: report
        log_fn(f"✅ Loaded {len(state_dict)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})")
        if missing:
            log_fn(f"   Missing: {missing[:3]}...")
        if unexpected:
            log_fn(f"   Unexpected: {unexpected[:3]}...")
        
        return True
        
    except Exception as e:
        log_fn(f"❌ Failed: {e}")
        return False


# The indices that the raw vector should be mapped to in the unified action vector
AGILEX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]


# Create the Video2Act model
def create_model(args, **kwargs):
    _sanitize_runtime_wan_paths(args)

    # Pass use_video_encoder_replacement to args if it's in kwargs
    if "use_video_encoder_replacement" in kwargs:
        args["use_video_encoder_replacement"] = kwargs["use_video_encoder_replacement"]

    return Video2ActRuntime(args, **kwargs)


class Video2ActRuntime(object):
    """
    Video2Act model wrapper for framepack-enhanced inference.
    The WAN feature extractor is handled by the Video2ActRunnerVideo2ActRealtime internally.
    """

    def __init__(
        self, args,
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained=None,
        pretrained_vision_encoder_name_or_path=None,
        anchor=None,
        **kwargs
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        self.anchor = anchor
        
        self.latents_buffer = None  # unused in dual-stream mode; kept for single-stream compat
        
        # Initialize lightweight Video2Act encoders (VAE + text) if paths are provided
        self.fp_encoder = None
        try:
            fp_cfg = self.args.get("framepack", {}) if isinstance(self.args, dict) else {}
            vae_path = fp_cfg.get("vae_path")
            text_encoder1_path = fp_cfg.get("text_encoder1_path")
            text_encoder2_path = fp_cfg.get("text_encoder2_path")
            tokenizer_path = fp_cfg.get("tokenizer_path")
            custom_system_prompt = fp_cfg.get("custom_system_prompt", "You are a helpful assistant.")
            guidance_scale = fp_cfg.get("guidance_scale", 1.0)
            vae_chunk_size = fp_cfg.get("vae_chunk_size")
            vae_spatial_tile_sample_min_size = fp_cfg.get("vae_spatial_tile_sample_min_size")

            if Video2ActLightweightEncoder is not None and (vae_path or text_encoder1_path):
                self.fp_encoder = Video2ActLightweightEncoder(
                    vae_path=vae_path,
                    text_encoder1_path=text_encoder1_path,
                    text_encoder2_path=text_encoder2_path,
                    tokenizer_path=tokenizer_path,
                    device=self.device,
                    custom_system_prompt=custom_system_prompt,
                    guidance_scale=guidance_scale,
                    load_vae=vae_path is not None,
                    load_text_encoders=text_encoder1_path is not None,
                )
                
        except Exception as e:
            # Keep running even if lightweight encoder failed to init
            print(f"Warning: Failed to initialize lightweight encoder: {e}")
            self.fp_encoder = None

        # Initialize vision encoder
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        
        # Initialize the Video2Act policy with Video2Act video integration (runner handles WAN feature extractor)
        self.policy = self.get_policy(pretrained, **kwargs)

        self._text_embed_cache = {}
        self._motion_latent_cache = []
        self.reset()

    def _build_framepack_batch_data(self, images, text_instruction=None, sparse_images=None, wan_text_embeds=None):
        """Build batch_data dict expected by runner when lightweight encoders are available.

        Produces keys: 'llama_vec', 'llama_attention_mask', 'clip_l_pooler',
        'image_latents' (from sparse_images if provided, otherwise single frame).

        Args:
            images: dense frames (t-1, t), 6 images, used for image embeddings
            text_instruction: language instruction
            sparse_images: sparse frames (9 frames x 3 cameras = 27 images) for VAE encode
            wan_text_embeds: path to a precomputed text-embedding .pkl; loaded directly if provided
        """
        if self.fp_encoder is None:
            return None

        batch_data = {}

        def to_rgb_array_for_vae(image):
            # Legacy precache compatibility.
            #
            # Existing WAN VAE latents were generated from historical RoboTwin
            # HDF5 files whose decoded arrays are numeric RGB, then
            # precache_wan22_latents.py applied cv2.COLOR_BGR2RGB. Numerically,
            # that means the cached VAE input is channel-flipped (BGR relative
            # to the original sim RGB). Keep the same flip online so raw/motion
            # VAE latents match the already-precached training latents.
            arr = np.array(image)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr[..., ::-1]
            return np.ascontiguousarray(arr)

        # If precomputed text embeddings are provided, load directly (cached after first load)
        if wan_text_embeds is not None:
            try:
                if wan_text_embeds not in self._text_embed_cache:
                    if os.path.exists(wan_text_embeds):
                        self._text_embed_cache[wan_text_embeds] = torch.load(wan_text_embeds, map_location='cpu')
                        print(f"✓ Loaded pre-computed text embeddings from {wan_text_embeds}")
                    else:
                        print(f"⚠ Warning: wan_text_embeds file not found: {wan_text_embeds}")
                text_embed_data = self._text_embed_cache.get(wan_text_embeds)
                if text_embed_data is not None:
                    # Precomputed file should contain: llama_vec, llama_attention_mask, clip_l_pooler
                    batch_data.update({
                        "llama_vec": text_embed_data["llama_vec"],
                        "llama_attention_mask": text_embed_data["llama_attention_mask"],
                        "clip_l_pooler": text_embed_data["clip_l_pooler"],
                    })
            except Exception as e:
                print(f"⚠ Warning: Failed to load wan_text_embeds: {e}")
        # Otherwise, if a text instruction is given and the text encoder is available, encode online
        elif text_instruction is not None and self.fp_encoder.text_encoder1 is not None:
            try:
                positive_ctx, _ = self.fp_encoder.encode_text(text_instruction)
                # Keep tensors on CPU; runner will move/dtype-cast
                batch_data.update({
                    "llama_vec": positive_ctx["llama_vec"],
                    "llama_attention_mask": positive_ctx["llama_attention_mask"],
                    "clip_l_pooler": positive_ctx["clip_l_pooler"],
                })
            except Exception:
                pass

        # Match training fallback: when WAN UMT5 text embeddings are unavailable,
        # provide zero placeholders so WAN conditioning keys always exist.
        if "llama_vec" not in batch_data:
            batch_data.update({
                "llama_vec": torch.zeros(1, 512, 4096),
                "llama_attention_mask": torch.zeros(1, 512),
                "clip_l_pooler": torch.zeros(1, 1, 1),
            })
            print("Warning: WAN text embeddings unavailable; using zero placeholders")

        if self.fp_encoder.vae is not None:
            # Raw stream: encode the most recent head-camera frame at [480,832]
            # → image_latents_head [1, C, 1, 30, 52]  (1×15×26 = 390 WAN tokens)
            raw_img = None
            if images is not None and len(images) >= 4:
                raw_img = images[3]  # dense images: [t-1×3cams, t×3cams], index 3 = head@t
            elif sparse_images is not None and len(sparse_images) >= 3:
                raw_img = sparse_images[-3]  # last timestep, cam_idx=0
            if raw_img is not None:
                arr = to_rgb_array_for_vae(raw_img)
                resized = self.fp_encoder.resize_image_to_bucket(arr, (832, 480))
                t_raw = torch.from_numpy(resized).float() / 127.5 - 1.0
                t_raw = t_raw.permute(2, 0, 1)[None, :, None].to(torch.float32)  # [1,3,1,480,832]
                batch_data["image_latents_head"] = self.fp_encoder.encode_image_with_vae(t_raw).to("cpu")

            # Motion stream: encode 61 consecutive head-camera frames at [224,224] as one video
            # VAE /4 temporal compression: 61 raw frames → [1, C, 16, 14, 14]
            # → 16×7×7 = 784 WAN tokens; stored as image_latents_motion
            if sparse_images is not None:
                num_frames = len(sparse_images) // 3  # should be 61
                mot_h, mot_w = 224, 224
                frame_tensors = []
                for t in range(num_frames):
                    head_img = sparse_images[t * 3 + 0]  # cam_idx=0 = head camera
                    arr = to_rgb_array_for_vae(head_img)
                    resized = self.fp_encoder.resize_image_to_bucket(arr, (mot_w, mot_h))
                    frame_tensors.append(
                        torch.from_numpy(resized).float() / 127.5 - 1.0
                    )  # [H, W, 3]
                # Stack to video: [1, 3, T, H, W]
                video = torch.stack(
                    [f.permute(2, 0, 1) for f in frame_tensors], dim=1
                )[None].to(torch.float32)  # [1, 3, 61, 224, 224]
                # VAE encode: [1, C, 16, 14, 14]  (61 frames → /4 → 16 latent frames)
                motion_latents = self.fp_encoder.encode_image_with_vae(video).to("cpu")
                mot_cfg = self.args.get("model", {}).get("dual_stream_framepack", {}).get("motion_stream", {})
                concat_chunks = int(mot_cfg.get("cache_concat_chunks", 1) or 1)
                if concat_chunks > 1:
                    self._motion_latent_cache.append(motion_latents)
                    self._motion_latent_cache = self._motion_latent_cache[-concat_chunks:]
                    chunks = list(self._motion_latent_cache)
                    while len(chunks) < concat_chunks:
                        chunks.insert(0, chunks[0])
                    batch_data["image_latents_motion"] = torch.cat(chunks, dim=2)
                else:
                    batch_data["image_latents_motion"] = motion_latents

        return batch_data if len(batch_data) > 0 else None

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        """Initialize vision encoder"""
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def get_policy(self, pretrained, **kwargs):
        """Initialize the Video2Act policy with Video2Act video integration"""
        # Calculate image condition length
        img_cond_len = (self.args["common"]["img_history_size"] 
                        * self.args["common"]["num_cameras"] 
                        * self.vision_model.num_patches)
        
        # Get framepack device settings
        video_opts = self.args["model"].get("video_device_settings", {})
        force_cpu = video_opts.get("force_cpu", False)
        
        # If force_cpu is enabled, override device settings
        if force_cpu:
            if "framepack" in self.args and "device" not in self.args["framepack"]:
                self.args["framepack"]["device"] = "cpu"
            print("🔧 Force CPU mode enabled for Video2Act (framepack model)")

        # Create WAN Video2Act policy. The merged class branches internally on
        # `use_video_token_compressor` (or legacy `use_dual_token_compressor`).
        from models.video2act_wan_policy import Video2ActWanPolicy as Video2ActRunner
        print("✅ Using Video2ActWanPolicy for eval")

        _model = Video2ActRunner(
            action_dim=self.args["common"]["state_dim"],
            pred_horizon=self.args["common"]["action_chunk_size"],
            config=self.args["model"],
            lang_token_dim=self.args["model"]["lang_token_dim"],
            img_token_dim=self.args["model"]["img_token_dim"],
            state_token_dim=self.args["model"]["state_token_dim"],
            video_encoder_token_dim=self.args["model"].get("video_encoder_token_dim", self.args["model"].get("framepack_encoder_token_dim")),
            max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                ("image", (self.args["common"]["img_history_size"],
                    self.args["common"]["num_cameras"],
                    -self.vision_model.num_patches)),
            ],
            lang_pos_embed_config=[
                ("lang", -self.args["dataset"]["tokenizer_max_length"]),
            ],
            video_config=self.args["framepack"],
            video_pos_embed_config=[
                ("video", -self.args["model"].get("video_compressed_tokens", self.args["model"].get("framepack_compressed_tokens"))),
            ],
            # Inference: WAN transformer placed on device immediately
            defer_wan_device_placement=False,
        )
        
        # Load pretrained weights if provided
        if pretrained is not None and pretrained != "":
            print(f"Loading pretrained weights from: {pretrained}")
            # Get checkpoint type from kwargs if available
            checkpoint_type = kwargs.get("checkpoint_type", None)
            success = load_pretrained_weights_simple(_model, pretrained, checkpoint_type=checkpoint_type)
            if success:
                print("✅ Pretrained weights loaded successfully")
            else:
                print("⚠️ Failed to load pretrained weights, using random initialization")
        
        # Decide whether to enable cache. YAML defaults can be overridden by
        # the VIDEO2ACT_CACHE_RATIO env var, which forces enable_cache=True
        # and sets cache_ratio to its integer value (set to 1 to disable).
        fp_cfg = self.args.get("framepack", {})
        enable_cache = fp_cfg.get("enable_cache", False)
        cache_ratio = int(fp_cfg.get("cache_ratio", 8))
        env_ratio = os.environ.get("VIDEO2ACT_CACHE_RATIO")
        if env_ratio is not None and env_ratio.strip() != "":
            cache_ratio = int(env_ratio)
            enable_cache = cache_ratio > 1

        if enable_cache:
            _model.enable_cache(cache_ratio=cache_ratio)
            print(f"✅ Cache enabled with ratio 1:{cache_ratio}")
        else:
            print("ℹ️  Cache disabled (enable_cache=False in config)")
        
        return _model

    def reset(self):
        """Set model to evaluation mode"""
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.vision_model.eval()
        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)
        
        # Clear latents buffer
        self.latents_buffer = None
        self._motion_latent_cache = []

    def _format_joint_to_state(self, joints):
        """Format joint proprioception into unified action vector"""
        joints = joints / torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            device=joints.device, dtype=joints.dtype
        )
        
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        state[:, :, AGILEX_STATE_INDICES] = joints
        
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint_backup(self, action):
        """[Backup] Original implementation: always extracts 14D joint angles via AGILEX_STATE_INDICES."""
        joints = action[:, :, AGILEX_STATE_INDICES]
        joints = joints * torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            device=joints.device, dtype=joints.dtype
        )
        return joints

    def _unformat_action_to_joint(self, action):
        """
        Unformat the unified action vector into the joint action to be executed.
        Matches the training data format in hdf5_vla_dataset.py.

        Args:
            action (torch.Tensor): unified action vector ([B, N, 128])

        Returns:
            joints (torch.Tensor): robot action.
                Single-arm Franka (right_arm_dim=0): [B, N, 10] = [x, y, z, ortho6d(6), gripper]
                Dual-arm: [B, N, 14] = joint angles (left6 + lgripper + right6 + rgripper)
        """
        arm_dim_cfg = self.args.get("arm_dim", {}) if isinstance(self.args, dict) else {}
        right_arm_dim = arm_dim_cfg.get("right_arm_dim", 7)
        left_arm_dim = arm_dim_cfg.get("left_arm_dim", 7)

        if right_arm_dim == 0:
            action_indices = [
                STATE_VEC_IDX_MAPPING["left_eef_pos_x"],
                STATE_VEC_IDX_MAPPING["left_eef_pos_y"],
                STATE_VEC_IDX_MAPPING["left_eef_pos_z"],
                STATE_VEC_IDX_MAPPING["left_eef_angle_0"],
                STATE_VEC_IDX_MAPPING["left_eef_angle_1"],
                STATE_VEC_IDX_MAPPING["left_eef_angle_2"],
                STATE_VEC_IDX_MAPPING["left_eef_angle_3"],
                STATE_VEC_IDX_MAPPING["left_eef_angle_4"],
                STATE_VEC_IDX_MAPPING["left_eef_angle_5"],
                STATE_VEC_IDX_MAPPING["left_gripper_open"],
            ]
            expected_dim = 10
        else:
            action_indices = AGILEX_STATE_INDICES
            expected_dim = left_arm_dim + 1 + right_arm_dim + 1

        joints = action[:, :, action_indices]
        joints = joints * torch.tensor(
            [[[1 for _ in range(expected_dim)]]],
            device=joints.device, dtype=joints.dtype
        )
        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds, text_instruction=None, use_video_features=True, sparse_images=None, wan_text_embeds=None,
             **kwargs):
        """Step function that delegates to the policy (runner handles WAN feature extractor)
        
        Args:
            proprio: robot state
            images: dense frames (t-1, t), 6 images for image embeddings
            text_embeds: text embeddings
            text_instruction: language instruction
            use_video_features: whether to use framepack features
            sparse_images: sparse frames (9 frames x 3 cameras = 27 images) for VAE encode
        """
        device = self.device
        dtype = self.dtype

        # Process images using the standard approach
        image_tensor = self._process_images(images)
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # Process proprioception
        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype) if text_embeds is not None else None

        # Optionally prepare batch_data for runner using lightweight encoders
        batch_data = self._build_framepack_batch_data(images, text_instruction, sparse_images, wan_text_embeds)

        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device) if text_embeds is not None else None,
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
            images=images,
            instruction_text=text_instruction,
            batch_data=batch_data,
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)
        return trajectory

    def _process_images(self, images):
        """Process images for vision encoder"""
        from PIL import Image
        from torchvision import transforms
        
        device = self.device
        dtype = self.dtype
        
        # Background image for padding
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        # Process each image
        image_tensor_list = []
        for image in images:
            if image is None:
                image = Image.fromarray(background_image)
            
            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)
            
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        return torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

    def cleanup(self):
        """Clean up resources (delegate to policy/runner)"""
        if hasattr(self.policy, 'cleanup'):
            self.policy.cleanup()
