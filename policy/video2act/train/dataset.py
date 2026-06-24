import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.hdf5_vla_dataset import HDF5VLADataset
from train.image_corrupt import image_corrupt

# WAN UMT5 fixed sequence length (must match precompute_wan_text_embeddings.py)
WAN_TEXT_SEQ_LEN = 512
WAN_TEXT_DIM = 4096


def get_clean_item(chunk_dir):
    dirty_bit = read_dirty_bit(chunk_dir)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()
            with open(file_path, "wb") as file:
                file.write(dirty_bit.tobytes())
            lock.release_lock()
            return
        except KeyboardInterrupt:
            lock.release_lock()
            raise
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_read_lock()
            with open(file_path, "rb") as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        except KeyboardInterrupt:
            lock.release_lock()
            raise
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")


class VLAConsumerDataset(Dataset):
    """
    VLA Dataset for Video2Act supervised training.

    Each sample contains:
      - T5 language embeddings  (precomp .pt files, used as Video2Act lang_tokens)
      - WAN UMT5 text embeddings (precomp .pkl files, used by Wan world model)
      - WAN22 image latents      (precomp HDF5 keys, used by Wan world model)
      - Camera images            (if not using latents)

    Batch keys produced by DataCollatorForVLAConsumerDataset:
      states               [B, 1, STATE_DIM]
      actions              [B, CHUNK, STATE_DIM]
      state_elem_mask      [B, 1, STATE_DIM]
      state_norm           [B, STATE_DIM]
      images               [B, num_cam*hist, 3, H, W]
      data_indices         [B]
      ctrl_freqs           [B]
      lang_embeds          [B, seq, 4096]   T5, for Video2Act transformer
      lang_attn_mask       [B, seq]         T5 attention mask
      llama_vec            [B, 1, 512, 4096]  WAN UMT5, for Wan world model
      llama_attention_mask [B, 1, 512]        WAN UMT5 mask
      clip_l_pooler        [B, 1, 1, 1]       WAN placeholder
      image_latents_head   [B, 1, C, 1, H, W]   WAN22 raw latent
      image_latents_motion [B, 1, C, 16, H, W]  WAN22 motion latent
    """

    def __init__(
        self,
        model_config_path,
        config,
        tokenizer,
        image_processor,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type="pretrain",
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        use_precomp_lang_embed=False,
        hdf5_dataset_class=None,
        dual_arm_dataset_class=None,
        ori_config=None,
    ):
        super().__init__()

        with open("configs/dataset_control_freq.json", "r") as fp:
            self.control_freq = json.load(fp)

        dataset_names_cfg = (
            "configs/pretrain_datasets.json"
            if dataset_type == "pretrain"
            else "configs/finetune_datasets.json"
        )
        with open(dataset_names_cfg, "r") as f:
            DATASET_NAMES = json.load(f)
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.image_processor = image_processor
        self.model_config_path = model_config_path
        self.buffer_dir = config["buf_path"]
        self.num_chunks = config["buf_num_chunks"]
        self.chunk_size = config["buf_chunk_size"]
        self.tokenizer_max_length = config["tokenizer_max_length"]
        self.image_aspect_ratio = config["image_aspect_ratio"]
        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob
        self.cam_ext_mask_prob = cam_ext_mask_prob
        self.use_hdf5 = use_hdf5
        self.use_precomp_lang_embed = use_precomp_lang_embed

        # T5 empty embedding for lang_tokens masking
        self.empty_lang_embed = None

        # WAN zero embeddings for WAN-path masking
        self.empty_llama_vec = torch.zeros(1, WAN_TEXT_SEQ_LEN, WAN_TEXT_DIM)
        self.empty_llama_mask = torch.zeros(1, WAN_TEXT_SEQ_LEN)
        self.empty_clip_pooler = torch.zeros(1, 1, 1)

        # HDF5 dataset
        self.hdf5_dataset = None
        if use_hdf5:
            if dual_arm_dataset_class is not None:
                self.hdf5_dataset = dual_arm_dataset_class(self.model_config_path)
            elif hdf5_dataset_class is not None:
                self.hdf5_dataset = hdf5_dataset_class(self.model_config_path)
            else:
                self.hdf5_dataset = HDF5VLADataset(
                    self.model_config_path, config=ori_config
                )

        with open("configs/dataset_stat.json", "r") as f:
            self.dataset_stat = json.load(f)

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug
        self.last_content = None
        self.last_meta = None

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    @staticmethod
    def _load_data_from_chunk(chunk_dir, chunk_item_idx):
        time_stmp = time.time()
        while time.time() - time_stmp < 10.0:
            try:
                locks = []
                file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "r") as file:
                    json_content = json.load(file)
                lock.release_lock()

                file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "rb") as file:
                    sample_dict = np.load(file)
                    meta = tuple(sample_dict.values())
                lock.release_lock()
                return json_content, meta
            except KeyboardInterrupt:
                for lk in locks:
                    lk.release_lock()
                raise
            except BaseException:
                for lk in locks:
                    lk.release_lock()
                continue
        raise RuntimeError("Failed to load sample.")

    def __len__(self):
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        return self.num_chunks * self.chunk_size

    def _safe_load(self, index):
        read_chunk_item_indices = []
        read_chunk_idx = index // self.chunk_size
        while not read_chunk_item_indices:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            try:
                read_chunk_item_indices = get_clean_item(read_chunk_dir)
            except BaseException as e:
                print("Error searching clean chunk:", e)
                traceback.print_exc()
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks

        read_chunk_item_index = read_chunk_item_indices[index % len(read_chunk_item_indices)]
        try:
            dirty_bit = read_dirty_bit(read_chunk_dir)
            dirty_bit[read_chunk_item_index] = 1
            save_dirty_bit(read_chunk_dir, dirty_bit)
        except BaseException as e:
            print("Error modifying dirty bit:", e)
            traceback.print_exc()

        try:
            content, meta = self._load_data_from_chunk(
                read_chunk_dir, read_chunk_item_index
            )
            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            print("Error loading sample:", e)
            traceback.print_exc()
            content, meta = self.last_content, self.last_meta

        return (content, *meta)

    def __getitem__(self, index):
        while True:
            data_dict = None
            try:
                # ---------------------------------------------------------
                # Load raw data from HDF5 or buffer
                # ---------------------------------------------------------
                if self.use_hdf5:
                    res = self.hdf5_dataset.get_item()
                    content = res["meta"]
                    states = res["state"]
                    actions = res["actions"]
                    state_elem_mask = res["state_indicator"]
                    image_metas = [
                        res["cam_high"],        res["cam_high_mask"],
                        res["cam_right_wrist"], res["cam_right_wrist_mask"],
                        res["cam_left_wrist"],  res["cam_left_wrist_mask"],
                    ]
                    state_std = res["state_std"]
                    state_mean = res["state_mean"]
                    state_norm = res["state_norm"]

                    # WAN22 raw image latent: [1, C, 1, H, W]
                    image_latents_head = res.get("image_latents_head", None)
                    # WAN22 motion latent: [1, C, 16, H, W], encoded from the
                    # 61-frame window ending at the sampled timestep.
                    image_latents_motion = res.get("image_latents_motion", None)

                    # WAN UMT5 text embeddings: [1, 512, 4096] / [1, 512] / [1, 1, 1]
                    wan_llama_vec = res.get("llama_vec", None)
                    wan_llama_mask = res.get("llama_attention_mask", None)
                    wan_clip_pooler = res.get("clip_l_pooler", None)
                else:
                    (
                        content, _,
                        states, _,
                        actions, _,
                        state_elem_mask,
                        *image_metas,
                        state_std, state_mean, state_norm,
                    ) = self._safe_load(index)
                    image_latents_head = None
                    image_latents_motion = None
                    wan_llama_vec = None
                    wan_llama_mask = None
                    wan_clip_pooler = None

                # ---------------------------------------------------------
                # Build data_dict
                # ---------------------------------------------------------
                data_dict = {}
                data_dict["dataset_name"] = content["dataset_name"]
                data_dict["data_idx"] = self.dataset_name2id[data_dict["dataset_name"]]
                data_dict["ctrl_freq"] = (
                    self.control_freq[data_dict["dataset_name"]]
                    if random.random() > self.cond_mask_prob else 0
                )

                # State noise
                if self.state_noise_snr is not None:
                    states = states + np.random.normal(
                        0.0,
                        state_std / np.sqrt(10 ** (self.state_noise_snr / 10)),
                        states.shape,
                    )

                ds_state_mean = np.array(
                    self.dataset_stat[data_dict["dataset_name"]]["state_mean"]
                )
                ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))

                data_dict["states"] = (
                    states if random.random() > self.cond_mask_prob else ds_state_mean
                )
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = (
                    state_elem_mask
                    if random.random() > self.cond_mask_prob
                    else np.zeros_like(state_elem_mask)
                )
                data_dict["state_norm"] = state_norm

                # WAN22 latents (always included when available; masking handled by WAN model)
                if image_latents_head is not None:
                    data_dict["image_latents_head"] = image_latents_head  # [1, C, 1, H, W]
                if image_latents_motion is not None:
                    data_dict["image_latents_motion"] = image_latents_motion  # [1, C, 16, H, W]

                # WAN UMT5 text embeddings with conditional masking
                if wan_llama_vec is not None:
                    if random.random() > self.cond_mask_prob:
                        data_dict["llama_vec"] = wan_llama_vec                # [1, 512, 4096]
                        data_dict["llama_attention_mask"] = wan_llama_mask    # [1, 512]
                        data_dict["clip_l_pooler"] = wan_clip_pooler          # [1, 1, 1]
                    else:
                        data_dict["llama_vec"] = self.empty_llama_vec
                        data_dict["llama_attention_mask"] = self.empty_llama_mask
                        data_dict["clip_l_pooler"] = self.empty_clip_pooler
                else:
                    # Fallback zeros so the model always finds these keys
                    data_dict["llama_vec"] = self.empty_llama_vec
                    data_dict["llama_attention_mask"] = self.empty_llama_mask
                    data_dict["clip_l_pooler"] = self.empty_clip_pooler

                # ---------------------------------------------------------
                # Camera images
                # ---------------------------------------------------------
                background_color = np.array(
                    [int(x * 255) for x in self.image_processor.image_mean],
                    dtype=np.uint8,
                ).reshape(1, 1, 3)
                background_image = (
                    np.ones(
                        (
                            self.image_processor.size["height"],
                            self.image_processor.size["width"],
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    * background_color
                )

                image_metas_pairs = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob

                rearranged = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = image_metas_pairs[j]
                        image, valid = images[i], image_mask[i]
                        if (
                            valid
                            and math.prod(image.shape) > 0
                            and random.random() > mask_probs[j]
                        ):
                            rearranged.append((image, True))
                        else:
                            rearranged.append((background_image.copy(), False))

                preprocessed_images = []
                proc = self.image_processor
                for image, valid in rearranged:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)

                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        avg_bright = (
                            sum(sum(p) for p in pixel_values)
                            / (len(pixel_values) * 255.0 * 3)
                        )
                        if avg_bright <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    if valid and self.image_aug and random.random() > 0.5:
                        aug_type = random.choice(["corrupt_only", "color_only", "both"])
                        if aug_type != "corrupt_only":
                            image = transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03
                            )(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                    if self.image_aspect_ratio == "pad":

                        def expand2square(pil_img, bg):
                            w, h = pil_img.size
                            if w == h:
                                return pil_img
                            side = max(w, h)
                            result = Image.new(pil_img.mode, (side, side), bg)
                            result.paste(
                                pil_img,
                                ((side - w) // 2, (side - h) // 2),
                            )
                            return result

                        image = expand2square(
                            image, tuple(int(x * 255) for x in proc.image_mean)
                        )
                    image = proc.preprocess(image, return_tensors="pt")["pixel_values"][0]
                    preprocessed_images.append(image)

                data_dict["images"] = preprocessed_images

                # ---------------------------------------------------------
                # T5 language embeddings (for Video2Act lang_tokens)
                # ---------------------------------------------------------
                if self.use_precomp_lang_embed:
                    # Strip trailing period so tokenizer lookups are consistent
                    instr = content["instruction"]
                    if instr.endswith("."):
                        instr = instr[:-1]
                    lang_embed = torch.load(instr, weights_only=False)
                    if random.random() > self.cond_mask_prob:
                        data_dict["lang_embed"] = lang_embed
                    else:
                        if (self.empty_lang_embed is None or self.empty_lang_embed.shape != lang_embed.shape or
                                self.empty_lang_embed.dtype != lang_embed.dtype):
                            self.empty_lang_embed = torch.zeros_like(lang_embed)
                        data_dict["lang_embed"] = self.empty_lang_embed
                else:
                    instruction = (
                        content["instruction"]
                        if random.random() > self.cond_mask_prob
                        else ""
                    )
                    data_dict["input_ids"] = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    ).input_ids[0]
                    assert len(data_dict["input_ids"]) <= self.tokenizer_max_length, (
                        f"Instruction length {len(data_dict['input_ids'])} exceeds "
                        f"max {self.tokenizer_max_length}."
                    )

                # ---------------------------------------------------------
                # Convert ndarray → tensor
                # ---------------------------------------------------------
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                return data_dict

            except BaseException as e:
                name = data_dict.get("dataset_name") if data_dict else "unknown"
                print(f"Error processing sample from {name}:", e)
                traceback.print_exc()
                index = (index + 1) % len(self)


class DataCollatorForVLAConsumerDataset:
    """
    Collates a list of VLAConsumerDataset samples into a batch.

    Batch keys:
        states               [B, 1, STATE_DIM]
        actions              [B, CHUNK, STATE_DIM]
        state_elem_mask      [B, 1, STATE_DIM]
        state_norm           [B, STATE_DIM]
        images               [B, num_cam*hist, 3, H, W]
        data_indices         [B]
        ctrl_freqs           [B]

        T5 path (precomp_lang_embed=True):
            lang_embeds      [B, seq, 4096]   padded T5
            lang_attn_mask   [B, seq]         bool
        tokenizer path:
            input_ids        [B, seq]         padded token ids
            lang_attn_mask   [B, seq]         bool

        WAN world-model keys (always present):
            llama_vec            [B, 1, 512, 4096]
            llama_attention_mask [B, 1, 512]
            clip_l_pooler        [B, 1, 1, 1]
            image_latents_head   [B, 1, C, 1, H, W]   raw stream (if available)
            image_latents_motion [B, 1, C, 16, H, W]  motion stream (if available)
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "state_norm": [],
            "images": [],
            "data_indices": [],
            "ctrl_freqs": [],
            # WAN world-model
            "llama_vec": [],
            "llama_attention_mask": [],
            "clip_l_pooler": [],
        }
        image_latents_head_list = []
        image_latents_motion_list = []
        input_ids = []
        lang_embeds = []
        lang_embed_lens = []

        for inst in instances:
            for key in ("states", "actions", "state_elem_mask", "state_norm"):
                v = inst[key]
                batch[key].append(
                    v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                )

            batch["images"].append(torch.stack(inst["images"], dim=0))
            batch["data_indices"].append(inst["data_idx"])
            batch["ctrl_freqs"].append(inst["ctrl_freq"])

            # WAN UMT5 text  [1, 512, 4096] / [1, 512] / [1, 1, 1]
            batch["llama_vec"].append(inst["llama_vec"])
            batch["llama_attention_mask"].append(inst["llama_attention_mask"])
            batch["clip_l_pooler"].append(inst["clip_l_pooler"])

            # WAN22 raw latent  [1, C, 1, H, W]  (optional)
            if "image_latents_head" in inst:
                image_latents_head_list.append(inst["image_latents_head"])
            # WAN22 motion latent [1, C, 16, H, W] (optional)
            if "image_latents_motion" in inst:
                image_latents_motion_list.append(inst["image_latents_motion"])

            # T5 / tokenizer
            if "input_ids" in inst:
                input_ids.append(inst["input_ids"])
            else:
                lang_embeds.append(inst["lang_embed"])
                lang_embed_lens.append(inst["lang_embed"].shape[0])

        # Stack scalar-list keys
        for key in ("states", "actions", "state_elem_mask", "state_norm", "images"):
            batch[key] = torch.stack(batch[key], dim=0)
        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])

        # WAN UMT5: stack → [B, 1, 512, 4096] etc.
        batch["llama_vec"] = torch.stack(batch["llama_vec"], dim=0)
        batch["llama_attention_mask"] = torch.stack(batch["llama_attention_mask"], dim=0)
        batch["clip_l_pooler"] = torch.stack(batch["clip_l_pooler"], dim=0)

        # WAN22 latents: stack → [B, 1, C, T, H, W]
        if image_latents_head_list:
            if len(image_latents_head_list) != len(instances):
                raise KeyError("Only part of the batch has image_latents_head.")
            batch["image_latents_head"] = torch.stack(image_latents_head_list, dim=0)
        if image_latents_motion_list:
            if len(image_latents_motion_list) != len(instances):
                raise KeyError("Only part of the batch has image_latents_motion.")
            batch["image_latents_motion"] = torch.stack(image_latents_motion_list, dim=0)

        # T5 / tokenizer language embeddings
        if input_ids:
            padded = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            batch["input_ids"] = padded
            batch["lang_attn_mask"] = padded.ne(self.tokenizer.pad_token_id)
        else:
            padded = torch.nn.utils.rnn.pad_sequence(
                lang_embeds, batch_first=True, padding_value=0.0
            )
            attn_mask = torch.zeros(
                padded.shape[0], padded.shape[1], dtype=torch.bool
            )
            for i, l in enumerate(lang_embed_lens):
                attn_mask[i, :l] = True
            batch["lang_embeds"] = padded
            batch["lang_attn_mask"] = attn_mask

        return batch
