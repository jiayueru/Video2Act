#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path

# get current workspace
current_file = Path(__file__)

import json
import sys

parent_dir = current_file.parent
sys.path.append(str(parent_dir))

import os

import argparse

import threading
import time
import yaml
from collections import deque

import numpy as np
import torch
from PIL import Image as PImage
import cv2

import sys, os

# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent, "models"))

from scripts.agilex_model import create_model
from multimodal_encoder.t5_encoder import T5Embedder
from config_utils import load_yaml_with_env

global_path = parent_dir.parent


class Video2Act:

    def __init__(
        self,
        pretrained_model_name_or_path,
        task_name,
        left_arm_dim,
        right_arm_dim,
        action_step,
        config_path=None,
        model_config_path=None,
        fixed_lang_embed=False,
        fixed_lang_embed_path=None,
        fixed_wan_text_embed_path=None,
    ):
        # set path
        current_file = Path(__file__)
        # global_path resolves to .../policy ; weights live at policy/weights/...
        self.global_path = current_file.parent.parent
        self._video2act_root = current_file.parent
        # config_path: from launch.json --config_path or default
        if config_path is not None:
            if os.path.isabs(config_path):
                self._config_path = config_path
            else:
                self._config_path = os.path.abspath(
                    os.path.join(self._video2act_root, config_path)
                )
        else:
            self._config_path = os.path.join(
                self._video2act_root, "configs/video2act_template.yaml"
            )
        self._model_config_path = self._resolve_optional_path(model_config_path)
        self.model_config = {}
        if self._model_config_path is not None and os.path.isfile(self._model_config_path):
            self.model_config = load_yaml_with_env(self._model_config_path)

        self.fixed_lang_embed = (
            self._as_bool(fixed_lang_embed)
            or self._as_bool(self.model_config.get("fixed_lang_embed", False))
        )
        self.fixed_lang_embed_path = self._resolve_optional_path(fixed_lang_embed_path)
        self.fixed_wan_text_embed_path = self._resolve_optional_path(fixed_wan_text_embed_path)
        # load the config
        self.config = {
            "episode_len": 10000,  # args.max_publish_step
            "state_dim": left_arm_dim + 1 + right_arm_dim +
            1,  # 14 dims action:[left joint angles,left gripper,right joint angles,right gripper]
            "chunk_size": 64,  # args.chunk_size
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        }
        self.args = {
            "max_publish_step": 10000,  # Maximum number of action publishing steps
            "seed": None,  # Random seed
            "ctrl_freq": 25,  # The control frequency of the robot
            "chunk_size": 64,  # Action chunk size
            "config_path": self._config_path,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
        }

        # Load Video2Act model
        self.left_arm_dim, self.right_arm_dim = left_arm_dim, right_arm_dim
        self.policy = self.make_policy(self.args)
        self.max_publish_step = self.config["episode_len"]
        self.chunk_size = self.config["chunk_size"]
        self.task_name = task_name
        self.observation_window = None
        self.img_size = (640, 480)
        self.action_step = action_step
        self.text_instruction = None
        self.tokenizer = None
        self.text_encoder = None
        config_yaml = load_yaml_with_env(self._config_path)
        self.sparse_img_history_size = config_yaml.get("common", {}).get("sparse_img_history_size", 9)
        self.sparse_img_interval = config_yaml.get("common", {}).get("sparse_img_interval", 5)
        self._init_fixed_language_paths()
        if self.fixed_lang_embed and self.fixed_lang_embed_path:
            print("fixed_lang_embed=True; skipping T5 text encoder load for eval")
        else:
            self.set_language_embed()

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def _resolve_optional_path(self, path):
        if path in (None, "", False):
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self._video2act_root, path))

    def _select_task_data_dir(self):
        data_path = self.model_config.get("data_path")
        if data_path is None:
            return None
        candidates = data_path if isinstance(data_path, list) else [data_path]
        candidates = [self._resolve_optional_path(p) for p in candidates if p]
        for path in candidates:
            if self.task_name and self.task_name in os.path.basename(path):
                return path
        return candidates[0] if candidates else None

    def _init_fixed_language_paths(self):
        if not self.fixed_lang_embed:
            return

        task_dir = self._select_task_data_dir()
        if task_dir is None:
            print("fixed_lang_embed=True but model_config.data_path is unavailable; using eval instruction text")
            return

        if self.fixed_lang_embed_path is None:
            fixed_embed = os.path.join(task_dir, "fixed_lang_embed.pt")
            if os.path.isfile(fixed_embed):
                self.fixed_lang_embed_path = fixed_embed
            else:
                instr_dir = os.path.join(task_dir, "episode_0", "instructions")
                if os.path.isdir(instr_dir):
                    pts = sorted(
                        os.path.join(instr_dir, fn)
                        for fn in os.listdir(instr_dir)
                        if fn.endswith(".pt")
                    )
                    if pts:
                        self.fixed_lang_embed_path = pts[0]

        if self.fixed_wan_text_embed_path is None:
            text_folder = (
                "unseen_text_embeddings"
                if self.model_config.get("use_unseen_text_embeddings", False)
                else "text_embeddings"
            )
            emb_dir = os.path.join(task_dir, text_folder)
            if os.path.isdir(emb_dir):
                pkls = sorted(
                    os.path.join(emb_dir, fn)
                    for fn in os.listdir(emb_dir)
                    if fn.endswith(".pkl")
                )
                if pkls:
                    self.fixed_wan_text_embed_path = pkls[0]

        if self.fixed_lang_embed_path:
            print(f"fixed Video2Act language embedding: {self.fixed_lang_embed_path}")
        if self.fixed_wan_text_embed_path:
            print(f"fixed WAN text embedding: {self.fixed_wan_text_embed_path}")

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language_embed(self):
        GPU = 0
        MODEL_PATH = os.environ.get(
            "TEXT_ENCODER_NAME",
            os.path.join(self.global_path, "weights/RDT/t5-v1_1-xxl"),
        )
        CONFIG_PATH = self._config_path
        config = load_yaml_with_env(CONFIG_PATH)
        device = torch.device(f"cuda:{GPU}")
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=device,
            use_offload_folder=None,
            local_files_only=os.path.isdir(MODEL_PATH),
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model
        self.text_encoder.eval()

    # set language randomly
    def random_set_language(self, instruction=None):
        assert instruction is not None, "Missing input instruction"
        self.set_language_instruction(instruction)

    # encoding language
    def set_language_instruction(self, language_instruction, save_dir=None, task_name=None):
        assert ((save_dir is None) ^ (task_name is None)) == False, "input error"

        if os.path.isfile(language_instruction):
            lang_obj = torch.load(language_instruction, map_location="cpu", weights_only=False)
            if isinstance(lang_obj, dict):
                print(f"Running with instruction: \"{lang_obj.get('instruction', '')}\" from \"{lang_obj.get('name', language_instruction)}\"")
                self.lang_embeddings = lang_obj["embeddings"]
            else:
                print(f"loading fixed instruction embedding from \"{language_instruction}\"")
                self.lang_embeddings = lang_obj
            if self.lang_embeddings.ndim == 2:
                self.lang_embeddings = self.lang_embeddings.unsqueeze(0)
            print("loading instruction from pre-embed path")
        else:
            if self.text_encoder is None or self.tokenizer is None:
                self.set_language_embed()
            device = next(self.text_encoder.parameters()).device
            with torch.no_grad():
                tokens = self.tokenizer(
                    language_instruction,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )["input_ids"].to(device)
                tokens = tokens.view(1, -1)
                output = self.text_encoder(tokens)
                pred = output.last_hidden_state.detach().cpu()

            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{task_name}.pt")
                torch.save({
                    "name": task_name,
                    "instruction": language_instruction,
                    "embeddings": pred,
                }, save_path)

            del tokens, output
            torch.cuda.empty_cache()
            self.lang_embeddings = pred
        self.text_instruction = language_instruction
        print(f"successfully set instruction: {language_instruction}")

    def set_eval_language_instruction(self, language_instruction):
        if self.fixed_lang_embed and self.fixed_lang_embed_path:
            self.set_language_instruction(self.fixed_lang_embed_path)
        else:
            self.set_language_instruction(language_instruction)


    # Update the observation window buffer (61 frames: 60 past + 1 current for motion stream)
    def update_observation_window(self, img_arr, state):
        def jpeg_mapping(img):
            # Legacy compatibility: img_arr from RoboTwin sim is numeric RGB.
            # Historical training HDF5 was encoded with cv2.imencode directly on
            # numeric RGB arrays, so cv2.imdecode returns the same numeric RGB
            # channel order. Keep this round-trip unchanged for SigLIP
            # train/eval consistency.
            if img is None:
                return None
            img = cv2.imencode(".jpg", img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img

        def resize_img(img, size):
            return cv2.resize(img, size)

        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = resize_img(img_front, self.img_size)
        img_left = resize_img(img_left, self.img_size)
        img_right = resize_img(img_right, self.img_size)
        img_front = jpeg_mapping(img_front)
        img_left = jpeg_mapping(img_left)
        img_right = jpeg_mapping(img_right)

        qpos = np.array(puppet_arm)
        device = (
            self.text_encoder.device
            if self.text_encoder is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        qpos = torch.from_numpy(qpos).float().to(device)

        current_obs = {
            "qpos": qpos,
            "images": {
                self.config["camera_names"][0]: img_front,
                self.config["camera_names"][1]: img_right,
                self.config["camera_names"][2]: img_left,
            },
        }

        if self.observation_window is None:
            self.observation_window = deque(maxlen=61)
            for _ in range(61):
                self.observation_window.append(current_obs)
        else:
            self.observation_window.append(current_obs)

    def get_action(self, img_arr=None, state=None, is_new_task=True, **kwargs):
        assert (img_arr is None) ^ (state is None) == False, "input error"
        if (img_arr is not None) and (state is not None):
            self.update_observation_window(img_arr, state)

        img_history_size = kwargs.get("img_history_size", self.sparse_img_history_size)
        img_interval = kwargs.get("img_interval", self.sparse_img_interval)
        with torch.inference_mode():
            action_buffer = inference_fn_with_history(
                self.config,
                self.policy,
                self.lang_embeddings,
                self.observation_window,
                self.text_instruction,
                is_new_task=is_new_task,
                img_history_size=img_history_size,
                img_interval=img_interval,
                wan_text_embeds=self.fixed_wan_text_embed_path,
                **kwargs,
            ).copy()

        return action_buffer

    def reset_obsrvationwindows(self):
        self.lang_embeddings = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    def _clear_slow_fast_cache(self):
        """Clear slow-fast cache to avoid cross-episode feature reuse."""
        # self.policy is the high-level wrapper; its inner runner is typically
        # stored in `self.policy.policy`.
        candidates = [self.policy, getattr(self.policy, "policy", None)]
        for obj in candidates:
            if obj is None:
                continue
            if hasattr(obj, "cached_adapted_hidden_states"):
                obj.cached_adapted_hidden_states = None
            if hasattr(obj, "cache_count"):
                obj.cache_count = 0

    # Initialize the model
    def make_policy(self, args):
        config_base_yaml = load_yaml_with_env(args["config_path"])
        args["config"] = config_base_yaml
        args["config"]["arm_dim"] = {
            "left_arm_dim": self.left_arm_dim,
            "right_arm_dim": self.right_arm_dim,
        }
        # pretrained_text_encoder_name_or_path = "weights/RDT/t5-v1_1-xxl"
        pretrained_vision_encoder_name_or_path = os.environ.get(
            "VISION_ENCODER_NAME",
            os.path.join(self.global_path, "weights/RDT/siglip-so400m-patch14-384"),
        )
        model = create_model(
            args=args["config"],
            dtype=torch.bfloat16,
            pretrained=args["pretrained_model_name_or_path"],
            # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            control_frequency=args["ctrl_freq"],
        )

        return model


# Video2Act inference (legacy 2-frame path; kept for compatibility)
def inference_fn(config, policy, lang_embeddings, observation_window, text_instruction=None, is_new_task=True, **kwargs):
    image_arrs = [
        observation_window[-2]["images"][config["camera_names"][0]],
        observation_window[-2]["images"][config["camera_names"][1]],
        observation_window[-2]["images"][config["camera_names"][2]],
        observation_window[-1]["images"][config["camera_names"][0]],
        observation_window[-1]["images"][config["camera_names"][1]],
        observation_window[-1]["images"][config["camera_names"][2]],
    ]
    images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
    proprio = observation_window[-1]["qpos"].unsqueeze(0)
    actions = (
        policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings,
            text_instruction=text_instruction,
            is_new_task=is_new_task,
            **kwargs,
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )
    return actions


def inference_fn_with_history(
    config,
    policy,
    lang_embeddings,
    observation_window,
    text_instruction=None,
    is_new_task=True,
    img_history_size=9,
    img_interval=5,
    **kwargs,
):
    """Inference with a 61-frame motion-stream window (training/eval consistent).

    Two image streams are produced:
    1. dense_images: 6 dense frames (t-1, t) for image embeddings
    2. sparse_images: all 61 consecutive frames feeding the motion-stream VAE
       (config: motion_stream.latent_num_frames=16, num_frames=61, temporal stride 4
       -> (61-1)//4+1 = 16 latent frames the runner expects).
    """
    window_len = len(observation_window)
    assert window_len == 61, f"observation window must have 61 frames, got {window_len}"

    dense_images = [
        PImage.fromarray(observation_window[idx]["images"][cam])
        if observation_window[idx]["images"][cam] is not None
        else None
        for idx in [-2, -1]
        for cam in config["camera_names"]
    ]

    # Motion stream VAE needs all 61 consecutive frames to produce the 16 latent
    # frames the runner expects. Sparse-sampled frames (9*interval=5) would only
    # yield 3 latent frames -> ValueError.
    sparse_images = [
        PImage.fromarray(observation_window[idx]["images"][cam])
        if observation_window[idx]["images"][cam] is not None
        else None
        for idx in range(len(observation_window))  # all 61 consecutive frames
        for cam in config["camera_names"]
    ]

    wan_text_embeds = kwargs.get("wan_text_embeds")

    return (
        policy.step(
            proprio=observation_window[-1]["qpos"].unsqueeze(0),
            images=dense_images,
            sparse_images=sparse_images,
            text_embeds=lang_embeddings,
            text_instruction=text_instruction,
            is_new_task=is_new_task,
            wan_text_embeds=wan_text_embeds,
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )
