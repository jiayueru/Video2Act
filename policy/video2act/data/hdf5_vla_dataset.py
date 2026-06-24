import os
import fnmatch
import random

import h5py
import yaml
import cv2
import numpy as np
import torch

from configs.state_vec import STATE_VEC_IDX_MAPPING


VIDEO2ACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# WAN UMT5 max sequence length (fixed pad target from precompute_wan_text_embeddings.py)
WAN_TEXT_SEQ_LEN = 512
WAN_TEXT_DIM = 4096


class HDF5VLADataset:
    """
    HDF5 dataset for Video2Act training.

    HDF5 per-episode structure:
        action                                (T, DOF)
        observations/qpos                     (T, DOF)
        observations/left_arm_dim             (1,)
        observations/right_arm_dim            (1,)
        observations/images/cam_high          (T,)  JPEG bytes
        observations/images/cam_left_wrist    (T,)  JPEG bytes
        observations/images/cam_right_wrist   (T,)  JPEG bytes
        observations/wan22_image_latents/head        (T, 1, C, 1, 30, 52)  float32  raw stream  [1f@480×832]
        observations/wan22_image_latents_motion/head (T, 1, C, 16, 14, 14) float32  motion stream [61f@224×224→VAE/4→16]
        observations/image_latents/head              (T, 1, C, 1, H, W)    float16  [legacy fallback]

    Task-directory structure:
        <task_dir>/
            text_embeddings/          <- WAN UMT5 pkl files (seen)
                seen_0000.pkl         <- {"llama_vec": [1,512,4096],
                                          "llama_attention_mask": [1,512],
                                          "clip_l_pooler": [1,1,1]}
                ...
            unseen_text_embeddings/   <- WAN UMT5 pkl files (unseen)
            episode_N/
                episode_N.hdf5
                instructions/         <- T5 .pt files for RDT lang conditioning
                    lang_embed_0.pt   <- Tensor [seq_len, 4096]
                    ...

    Returns per sample (all tensors on CPU):
        meta                dict
        state               ndarray [1, STATE_DIM]
        state_std           ndarray [STATE_DIM]
        state_mean          ndarray [STATE_DIM]
        state_norm          ndarray [STATE_DIM]
        actions             ndarray [CHUNK_SIZE, STATE_DIM]
        state_indicator     ndarray [STATE_DIM]
        cam_high            ndarray [IMG_HISTORY_SIZE, H, W, 3]
        cam_high_mask       ndarray [IMG_HISTORY_SIZE]  bool
        cam_left_wrist      ndarray  (same)
        cam_left_wrist_mask ndarray
        cam_right_wrist     ndarray
        cam_right_wrist_mask ndarray
        image_latents_head    Tensor  [1, C, 1, 30, 52]   raw stream (if available)
        image_latents_motion  Tensor  [1, C, 16, 14, 14]  motion stream (if available)
        llama_vec           Tensor  [1, 512, 4096]     (WAN UMT5 text, if available)
        llama_attention_mask Tensor [1, 512]           (WAN UMT5 mask, if available)
        clip_l_pooler       Tensor  [1, 1, 1]          (WAN placeholder, if available)
    """

    # Priority order when looking for cached latents
    LATENT_GROUP_PRIORITY = ["wan22_image_latents", "image_latents"]

    def __init__(self, model_config_path, config=None, use_latents=True):
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # data_path may be a single string or a list of strings
        _raw_path = model_config["data_path"]
        HDF5_DIRS = _raw_path if isinstance(_raw_path, list) else [_raw_path]
        self.DATASET_NAME = "agilex"
        self.use_latents = use_latents

        # text_embeddings/ vs unseen_text_embeddings/
        self.use_unseen_text = model_config.get("use_unseen_text_embeddings", False)
        self.text_embeddings_folder = (
            "unseen_text_embeddings" if self.use_unseen_text else "text_embeddings"
        )
        # If True, always use the first (sorted) pkl / .pt instead of random sampling.
        self.fixed_lang_embed = model_config.get("fixed_lang_embed", False)

        # Resolve base config
        if config is None:
            cfg_path = model_config.get("config_path", "configs/video2act_template.yaml")
            with open(cfg_path, "r") as f:
                config = yaml.safe_load(f)

        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISTORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]
        # Latent temporal window (default: current frame only)
        self.LATENT_WINDOW_SIZE_PAST = config["common"].get("latent_window_size_past", 0)
        self.LATENT_WINDOW_SIZE_FUTURE = config["common"].get("latent_window_size_future", 0)
        self.LATENT_INTERVAL = config["common"].get("latent_interval", 1)
        motion_cfg = (
            config.get("model", {})
            .get("dual_stream_framepack", {})
            .get("motion_stream", {})
        )
        self.MOTION_CACHE_CONCAT_CHUNKS = int(motion_cfg.get("cache_concat_chunks", 1))
        self.MOTION_CACHE_CONCAT_STRIDE = int(motion_cfg.get("cache_concat_stride", 61))

        # Collect all .hdf5 files across all data_path entries
        self.file_paths = []
        for hdf5_dir in HDF5_DIRS:
            for root, _, files in os.walk(hdf5_dir):
                for fn in fnmatch.filter(files, "*.hdf5"):
                    self.file_paths.append(os.path.join(root, fn))

        # Per-episode sampling weights proportional to length
        episode_lens = []
        for fp in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(fp)
            episode_lens.append(res["state"].shape[0] if valid else 0)
        total = float(sum(episode_lens)) or 1.0
        self.episode_sample_weights = np.array(episode_lens, dtype=float) / total

        # Pre-compute fixed instruction path per task dir:
        # always episode_0's lang_embed_0.pt (first episode dir, first .pt, both sorted).
        self._fixed_instr_per_task: dict = {}
        if self.fixed_lang_embed:
            task_dirs = {os.path.dirname(os.path.dirname(fp)) for fp in self.file_paths}
            for td in task_dirs:
                fixed = os.path.join(td, "fixed_lang_embed.pt")
                if not os.path.isfile(fixed):
                    fixed = ""
                    instr_d = os.path.join(td, "episode_0", "instructions")
                    if os.path.exists(instr_d):
                        pts = sorted([
                            os.path.join(instr_d, fn)
                            for fn in os.listdir(instr_d)
                            if fn.endswith(".pt")
                        ])
                        if pts:
                            fixed = pts[0]
                self._fixed_instr_per_task[td] = fixed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index=None, state_only=False):
        """Sample one training example (random episode if index is None)."""
        while True:
            fp = (np.random.choice(self.file_paths, p=self.episode_sample_weights)
                  if index is None else self.file_paths[index])
            valid, sample = (self.parse_hdf5_file(fp) if not state_only
                             else self.parse_hdf5_file_state_only(fp))
            if valid:
                return sample
            index = np.random.randint(0, len(self.file_paths))

    # ------------------------------------------------------------------
    # WAN latent helpers
    # ------------------------------------------------------------------

    def _select_latents_group(self, f):
        """Return the first matching latent group from LATENT_GROUP_PRIORITY, or None."""
        obs = f.get("observations")
        if obs is None:
            return None
        for name in self.LATENT_GROUP_PRIORITY:
            if name in obs and "head" in obs[name]:
                return obs[name]
        return None

    def _parse_image_latents(self, f, step_id):
        """
        Load raw-stream latent for the current step only.

        HDF5 stored per step: (1, C, 1, H_r, W_r)  [wan22_image_latents/head]
        Output: {"head": Tensor [1, C, 1, H_r, W_r]}
        """
        grp = self._select_latents_group(f)
        if grp is None:
            return {}

        ds = grp["head"]
        frame = torch.from_numpy(ds[step_id]).float()  # (1, C, 1, H, W) or (C, 1, H, W)

        # Normalise to [C, 1, H, W]
        if frame.ndim == 5:   # (1, C, 1, H, W)
            frame = frame[0]
        # frame: [C, 1, H, W] → unsqueeze batch → [1, C, 1, H, W]
        latent = frame.unsqueeze(0)
        return {"head": latent}

    def _parse_motion_latents(self, f, step_id):
        """
        Load motion-stream latent for the current step.

        HDF5 stored per step: (1, C, 16, H_m, W_m)  [wan22_image_latents_motion/head]
        Precomputed from 61 raw frames → VAE /4 temporal compression.
        Output: {"motion": Tensor [1, C, 16, H_m, W_m]}
        """
        obs = f.get("observations")
        if obs is None:
            return {}
        grp = obs.get("wan22_image_latents_motion")
        if grp is None or "head" not in grp:
            return {}

        ds = grp["head"]
        chunks = []
        # Older chunks first, current chunk last. For cache_concat_chunks=2 and
        # stride=61 this approximates a 32-latent history without new precache.
        for chunk_id in range(self.MOTION_CACHE_CONCAT_CHUNKS - 1, -1, -1):
            idx = max(0, step_id - chunk_id * self.MOTION_CACHE_CONCAT_STRIDE)
            frame = torch.from_numpy(ds[idx]).float()  # (1, C, 16, H, W) or (C, 16, H, W)
            if frame.ndim == 5:   # (1, C, 16, H, W)
                frame = frame[0]
            chunks.append(frame)
        motion = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        # motion: [C, T, H, W] → [1, C, T, H, W]
        return {"motion": motion.unsqueeze(0)}

    # ------------------------------------------------------------------
    # WAN UMT5 text embedding helpers
    # ------------------------------------------------------------------

    def _load_wan_text_embedding(self, task_dir):
        """
        Randomly pick one WAN UMT5 pkl from task_dir/text_embeddings/.

        pkl keys:
            llama_vec             Tensor [1, 512, 4096]
            llama_attention_mask  Tensor [1, 512]
            clip_l_pooler         Tensor [1, 1, 1]   (zero placeholder)

        Returns tensors with the same shapes (CPU), or (None, None, None) if unavailable.
        """
        emb_dir = os.path.join(task_dir, self.text_embeddings_folder)
        if not os.path.exists(emb_dir):
            return None, None, None

        pkls = [os.path.join(emb_dir, fn)
                for fn in os.listdir(emb_dir) if fn.endswith(".pkl")]
        if not pkls:
            return None, None, None

        chosen_pkl = sorted(pkls)[0] if self.fixed_lang_embed else random.choice(pkls)
        data = torch.load(chosen_pkl, map_location="cpu")

        def _t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return x.cpu() if x.is_cuda else x

        return _t(data["llama_vec"]), _t(data["llama_attention_mask"]), _t(data["clip_l_pooler"])

    # ------------------------------------------------------------------
    # HDF5 parsing
    # ------------------------------------------------------------------

    def parse_hdf5_file(self, file_path):
        """
        Parse one episode, sample a random timestep, return a training dict.
        WAN text embeddings are loaded outside the h5py context (task-level pkl).
        """
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]         # (T, DOF)
            left_arm_dim = f["observations"]["left_arm_dim"][:]
            right_arm_dim = f["observations"]["right_arm_dim"][:]
            num_steps = qpos.shape[0]

            # Skip leading still frames
            EPS = 1e-2
            indices = np.where(np.any(np.abs(qpos - qpos[0:1]) > EPS, axis=1))[0]
            if len(indices) == 0:
                raise ValueError(f"No qpos exceeds threshold in {file_path}")
            first_idx = indices[0]

            step_id = np.random.randint(first_idx - 1, num_steps)

            # Per-episode T5 instruction path (used as Video2Act lang_tokens source)
            ep_dir = os.path.dirname(file_path)
            instr_dir = os.path.join(ep_dir, "instructions")
            instr_names = (
                [os.path.join(instr_dir, fn)
                 for fn in os.listdir(instr_dir) if fn.endswith(".pt")]
                if os.path.exists(instr_dir) else []
            )
            if self.fixed_lang_embed:
                task_dir_key = os.path.dirname(ep_dir)
                instruction = self._fixed_instr_per_task.get(task_dir_key, "")
            elif instr_names:
                instruction = random.choice(instr_names)
            else:
                instruction = ""

            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
            }

            # State & action
            dof = left_arm_dim[0] + 1 + right_arm_dim[0] + 1
            scale = np.ones((1, dof))
            qpos_sc = qpos / scale
            target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE] / scale

            state = qpos_sc[step_id:step_id + 1]
            state_std = np.std(qpos_sc, axis=0)
            state_mean = np.mean(qpos_sc, axis=0)
            state_norm = np.sqrt(np.mean(qpos_sc ** 2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                actions = np.concatenate(
                    [actions,
                     np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))],
                    axis=0,
                )

            def fill_in_state(values):
                uni_idx = (
                    [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                     for i in range(left_arm_dim[0])]
                    + [STATE_VEC_IDX_MAPPING["left_gripper_open"]]
                    + [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                       for i in range(right_arm_dim[0])]
                    + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
                )
                out = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                out[..., uni_idx] = values
                return out

            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_state(actions)

            # Images: keep the legacy RoboTwin/RDT_wan color path for checkpoint
            # and precached-VAE compatibility.
            #
            # Historical RoboTwin HDF5 files were written from RGB arrays with
            # cv2.imencode(".jpg", rgb_array). OpenCV treats that array as BGR,
            # so cv2.imdecode below returns an array whose numeric channel order
            # matches the original RGB array. Do not apply BGR2RGB here:
            #   - SigLIP train sees numeric RGB, matching sim eval's jpeg_mapping.
            #   - WAN VAE latents are already precached with the legacy extra
            #     channel flip; online eval mirrors that in scripts/agilex_model.py.
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISTORY_SIZE + 1, 0), step_id + 1):
                    bits = f["observations"]["images"][key][i]
                    img = cv2.imdecode(np.frombuffer(bits, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"Failed to decode JPEG image {key}[{i}] in {file_path}")
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                    imgs = np.concatenate(
                        [np.tile(imgs[:1], (self.IMG_HISTORY_SIZE - imgs.shape[0], 1, 1, 1)),
                         imgs],
                        axis=0,
                    )
                return imgs

            cam_high = parse_img("cam_high")
            cam_left_wrist = parse_img("cam_left_wrist")
            cam_right_wrist = parse_img("cam_right_wrist")

            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISTORY_SIZE)
            cam_mask = np.array(
                [False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len
            )

            # Dual-stream latents
            image_latents = {}
            if self.use_latents:
                image_latents.update(self._parse_image_latents(f, step_id))
                image_latents.update(self._parse_motion_latents(f, step_id))

        # Assemble sample (h5py file is now closed)
        sample = {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_mask.copy(),
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_mask.copy(),
        }
        for view_name, latent in image_latents.items():
            # key: "image_latents_head", shape: [1, C, T, H, W]
            sample[f"image_latents_{view_name}"] = latent

        # WAN UMT5 text embeddings (task-level, loaded after closing hdf5)
        # ep_dir = <task_dir>/episode_N/  =>  task_dir = ep_dir's parent
        task_dir = os.path.dirname(ep_dir)
        llama_vec, llama_mask, clip_pooler = self._load_wan_text_embedding(task_dir)
        if llama_vec is not None:
            # Keep leading batch-dim=1 so collator can simply torch.stack
            sample["llama_vec"] = llama_vec             # [1, 512, 4096]
            sample["llama_attention_mask"] = llama_mask  # [1, 512]
            sample["clip_l_pooler"] = clip_pooler        # [1, 1, 1]

        return True, sample

    def parse_hdf5_file_state_only(self, file_path):
        """Parse full state/action trajectory (used for episode-length sampling weights)."""
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            left_arm_dim = f["observations"]["left_arm_dim"][:]
            right_arm_dim = f["observations"]["right_arm_dim"][:]

            indices = np.where(np.any(np.abs(qpos - qpos[0:1]) > 1e-2, axis=1))[0]
            if len(indices) == 0:
                raise ValueError(f"No qpos exceeds threshold in {file_path}")
            first_idx = indices[0]

            dof = left_arm_dim[0] + right_arm_dim[0] + 2
            scale = np.ones((1, dof))
            qpos = qpos / scale
            target_qpos = f["action"][:] / scale

            def fill_in_state(values):
                uni_idx = (
                    [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                     for i in range(left_arm_dim[0])]
                    + [STATE_VEC_IDX_MAPPING["left_gripper_open"]]
                    + [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                       for i in range(right_arm_dim[0])]
                    + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
                )
                out = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                out[..., uni_idx] = values
                return out

            return True, {
                "state": fill_in_state(qpos[first_idx - 1:]),
                "action": fill_in_state(target_qpos[first_idx - 1:]),
            }
