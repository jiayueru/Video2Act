#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import logging
import math
import os
import sys
import multiprocessing as mp
from pathlib import Path

# Add policy/video2act to Python path
current_dir = Path(__file__).resolve().parent.parent  # policy/video2act directory
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Use spawn for multiprocessing to avoid CUDA fork issues
mp.set_start_method('spawn', force=True)

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from safetensors.torch import load_model

from models.ema_model import EMAModel
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.video2act_wan_policy import Video2ActWanPolicy as Video2ActRunner
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res
from config_utils import load_yaml_with_env


if is_wandb_available():
    import wandb

def load_pretrained_weights_simple(model, pretrained_path, logger=None):
    """Non-strict weight loader supporting either a file or a directory.

    Args:
        model: freshly initialized model
        pretrained_path: pretrained weight path (file or directory)
        logger: optional logger

    Returns:
        bool: whether loading succeeded
    """
    log_fn = logger.info if logger else print

    try:
        # Directory: load pytorch_model.bin from inside
        if os.path.isdir(pretrained_path):
            model_path = os.path.join(pretrained_path, "pytorch_model.bin")
            log_fn(f"Loading weights from: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"pytorch_model.bin not found in {pretrained_path}")
                
            checkpoint = torch.load(model_path, map_location='cpu')
            pretrained_state_dict = checkpoint
            
        # File: load directly
        elif os.path.isfile(pretrained_path):
            log_fn(f"Loading weights from file: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'module' in checkpoint:
                    pretrained_state_dict = checkpoint['module']
                elif 'state_dict' in checkpoint:
                    pretrained_state_dict = checkpoint['state_dict']
                else:
                    pretrained_state_dict = checkpoint
            else:
                pretrained_state_dict = checkpoint
        else:
            raise FileNotFoundError(f"Path not found: {pretrained_path}")
        
        # Non-strict load
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        if missing_keys:
            log_fn(f"Missing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:
                log_fn(f"  - {key}")
            if len(missing_keys) > 10:
                log_fn(f"  ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            log_fn(f"Unexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:
                log_fn(f"  - {key}")
            if len(unexpected_keys) > 10:
                log_fn(f"  ... and {len(unexpected_keys) - 10} more")

        # motion_adaptor-related missing keys
        motion_keys = [k for k in missing_keys if 'motion_adaptor' in k or 'motion_adaptors' in k]
        if motion_keys:
            log_fn(f"Motion-related missing keys ({len(motion_keys)}):")
            for key in motion_keys:
                log_fn(f"  - {key}")
        
        log_fn(f"✅ Successfully loaded weights with {len(missing_keys)} missing keys")
        return True
        
    except Exception as e:
        error_msg = f"❌ Failed to load weights: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return False

def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: mitd
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- video2act
---
    """
    model_card = f"""
# Video2Act - {repo_id}

This is a Video2Act model derived from {base_model}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    # Read the config
    config = load_yaml_with_env(args.config_path)
        
    model_config = load_yaml_with_env(args.model_config_path)
    # print(model_config)
    output_dir = getattr(args, "output_dir", None) or model_config["checkpoint_path"]
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(from_pretrained=args.pretrained_text_encoder_name_or_path, 
                                model_max_length=config["dataset"]["tokenizer_max_length"], device=accelerator.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor

    # Always construct model from config first (to support new motion adapter structure)
    logger.info("Constructing model from provided config.")
    # Calculate the image condition length
    img_cond_len = (config["common"]["img_history_size"] 
                    * config["common"]["num_cameras"] 
                    * vision_encoder.num_patches)
    
    # Load Video2Act device settings from model config
    video_opts = model_config.get("video_device_settings", {})
    use_last_gpu = video_opts.get("use_last_gpu", True)

    # Set WAN device based on local rank so it loads directly to the right GPU
    wan_device = f"cuda:{accelerator.local_process_index}"
    config["framepack"]["device"] = wan_device

    logger.info(f"🚀 Video2Act Device Settings:")
    logger.info(f"  - Use Last GPU: {use_last_gpu}")
    logger.info(f"  - WAN device: {wan_device} (process {accelerator.local_process_index})")
    
    policy_model = Video2ActRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        video_encoder_token_dim=config["model"].get("video_encoder_token_dim", config["model"].get("framepack_encoder_token_dim")),
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=img_cond_len,
        img_pos_embed_config=[
            # No initial pos embed in the last grid size
            # since we've already done in ViT
            ("image", (config["common"]["img_history_size"], 
                config["common"]["num_cameras"], 
                -vision_encoder.num_patches)),  
        ],
        lang_pos_embed_config=[
            # Similarly, no initial pos embed for language
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        video_config=config["framepack"],
        video_pos_embed_config=[
            ("video", -config["model"].get("video_compressed_tokens", config["model"].get("framepack_compressed_tokens"))),
        ],
        use_video_compression=config["model"].get("use_video_compression", config["model"].get("use_framepack_compression", True)),
        video_compressed_dim=config["model"].get("video_compressed_dim", config["model"].get("framepack_compressed_dim", 3072)),
        video_compressed_tokens=config["model"].get("video_compressed_tokens", config["model"].get("framepack_compressed_tokens")),
        token_compressor_depth=config["model"].get("token_compressor_depth", 0),
        token_compressor_heads=config["model"].get("token_compressor_heads", 8),
        # Video2Act device parameters
        video_use_last_gpu=use_last_gpu,
        defer_wan_device_placement=False,  # Load WAN directly to target GPU
    )
    
    # Load pretrained weights if specified (compatible loading)
    if args.pretrained_model_name_or_path != "":
        logger.info(f"Loading pretrained weights from: {args.pretrained_model_name_or_path}")
        success = load_pretrained_weights_simple(policy_model, args.pretrained_model_name_or_path, logger)
        if success:
            logger.info("✅ Pretrained weights loaded successfully with new motion adapter structure")
        else:
            logger.warning("⚠️  Failed to load pretrained weights, using random initialization")
            
        
                                                                       
    # EMA: enabled by default
    ema_enabled = config["model"]["ema"].get("enabled", True)  # Default to True for better model performance
    ema_policy_model = None
    ema_model = None
    
    if ema_enabled:
        logger.info("✅ EMA is enabled for improved model stability and performance")
        # Create EMA model (will be moved to GPU after accelerator.prepare)
        ema_policy_model = copy.deepcopy(policy_model)
        ema_model = EMAModel(
            ema_policy_model,
            update_after_step=config["model"]["ema"]["update_after_step"],
            inv_gamma=config["model"]["ema"]["inv_gamma"],
            power=config["model"]["ema"]["power"],
            min_value=config["model"]["ema"]["min_value"],
            max_value=config["model"]["ema"]["max_value"]
        )
        logger.info("✅ EMA model created successfully")
    else:
        logger.info("⚠️ EMA is disabled - training without exponential moving average")

    # Note: Model saving now handled automatically via custom state_dict() method in Video2ActWanPolicy
    # The WAN feature extractor is automatically excluded when state_dict() is called
    
    if args.gradient_checkpointing:
        # TODO: 
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation - only optimize trainable parameters (excludes frozen framepack generator)
    params_to_optimize = policy_model.get_trainable_parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Dataset and DataLoaders creation:
    # For real-time framepack training, we use the original dataset without precomputed video features                                                           
    train_dataset = VLAConsumerDataset(
        model_config_path = args.model_config_path,# TODO
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        ori_config=config,
        # use_video_features=False,  # Always use original dataset without cached video files
    )
    sample_dataset = VLAConsumerDataset(
        model_config_path = args.model_config_path,# TODO
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        ori_config=config,
        # use_video_features=False,  # Always use original dataset without cached video files
    )                              
    
    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)                                                        
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    policy_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler
    )
    
    unwrapped_policy_model = accelerator.unwrap_model(policy_model)
    
    # Move EMA model to GPU
    if ema_enabled and ema_policy_model is not None:
        ema_policy_model.to(accelerator.device, dtype=weight_dtype)
        logger.info(f"✅ EMA model moved to {accelerator.device}")
    
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Use env-var project name, fallback to default
        wandb_project = os.environ.get("WANDB_PROJECT", "VLA")
        wandb_name = os.environ.get("WANDB_NAME", "RoboTwin_Video2Act")
        wandb_entity = os.environ.get("WANDB_ENTITY")
        wandb_kw = {"name": f"{wandb_name}"}
        if wandb_entity:
            wandb_kw["entity"] = wandb_entity
        accelerator.init_trackers(wandb_project, config=vars(args), init_kwargs={"wandb": wandb_kw})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Using WAN video-token features: {args.use_realtime_framepack_features}")
    logger.info("  Dataset: Using original dataset without precomputed cached video files")
    logger.info(f"  EMA enabled: {ema_enabled} (default: True)")
    if ema_enabled:
        logger.info(f"  EMA model storage: CPU (main process only)")
        logger.info(f"  EMA improves model stability and final performance")
    logger.info(f"  Multi-GPU setup: {accelerator.num_processes} processes")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path)) # load_module_strict=False
            except Exception as e:
                # load deepspeed's state_dict
                logger.info(f"Resuming training state failed: {e}. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                policy_model.module.load_state_dict(checkpoint["module"], strict=False)
                
            # Load EMA model
            if ema_enabled and ema_policy_model is not None:
                ema_path = os.path.join(args.output_dir, path, "ema", "model.safetensors")
                if os.path.exists(ema_path):
                    load_model(ema_policy_model, ema_path)
                    logger.info(f"✅ Loaded EMA model from {ema_path}")
                else:
                    logger.warning(f"⚠️ EMA checkpoint not found at {ema_path}")
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss_for_log = {}
    for epoch in range(first_epoch, args.num_train_epochs):

        policy_model.train()
        
        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        # Forward and backward...
        for batch in train_dataloader:
            with accelerator.accumulate(policy_model):
                images = batch["images"].to(dtype=weight_dtype) #[32, 6, 3, 384, 384]
                states = batch["states"].to(dtype=weight_dtype) # (B, T, D_a) [32, 1, 128]
                # We only use the last state as input
                states = states[:, -1:, :]
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]
                
                # Since we're using original dataset without cached video files,
                # video features will be extracted in real-time by the model
                batch_size = images.shape[0]
                
                with torch.no_grad():
                    batch_size, _, C, H, W = images.shape
                    flat_images = images.reshape(-1, C, H, W)
                    vision_chunk_size = int(os.environ.get("VISION_ENCODER_CHUNK_SIZE", "8"))
                    if vision_chunk_size > 0 and flat_images.shape[0] > vision_chunk_size:
                        image_embeds = torch.cat(
                            [
                                vision_encoder(flat_images[i:i + vision_chunk_size]).detach()
                                for i in range(0, flat_images.shape[0], vision_chunk_size)
                            ],
                            dim=0,
                        )
                    else:
                        image_embeds = vision_encoder(flat_images).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
                    # print(f"image_embeds shape: {image_embeds.shape}")
                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = batch["lang_embeds"].to(dtype=weight_dtype) \
                        if args.precomp_lang_embed \
                        else text_encoder(
                            input_ids=batch["input_ids"],
                            attention_mask=lang_attn_mask
                        )["last_hidden_state"].detach()
                
                state_elem_mask = state_elem_mask.unsqueeze(1)
                
                loss = policy_model(
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                        images=images,  # Pass images for real-time video feature extraction
                        batch_data=batch,
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = accelerator.unwrap_model(policy_model).get_trainable_parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            
            # EMA update on main process only (on GPU)
            if ema_enabled and accelerator.is_main_process and ema_model is not None:
                unwrapped_model = accelerator.unwrap_model(policy_model)
                ema_model.step(unwrapped_model)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    
                    # Use accelerator.save_state but override model file after saving
                    accelerator.save_state(save_path, safe_serialization=False)
                    
                    # Override the model state with filtered version (WAN feature extractor excluded)
                    model_checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
                    model_save_path = os.path.join(save_path, "pytorch_model")
                    if os.path.exists(model_save_path):
                        unwrapped_model = accelerator.unwrap_model(policy_model)
                        filtered_state_dict = unwrapped_model.state_dict()
                        model_checkpoint_path = os.path.join(model_save_path, "mp_rank_00_model_states.pt")
                        
                        # Save only the filtered model state, overriding DeepSpeed's version
                        torch.save({"module": filtered_state_dict}, model_checkpoint_path)
                        
                        # Check file size to confirm filtering worked
                        file_size_gb = os.path.getsize(model_checkpoint_path) / (1024**3)
                        logger.info(f"✅ Overrode model state with filtered version: {model_checkpoint_path}")
                        logger.info(f"📊 Filtered model file size: {file_size_gb:.2f} GB")
                        
                        if file_size_gb > 5:  # Still too large
                            logger.warning(f"⚠️ Model file still large ({file_size_gb:.2f} GB), WAN feature extractor may not be filtered")
                        else:
                            logger.info(f"✅ Model file size reduced successfully ({file_size_gb:.2f} GB)")
                    
                    # Save EMA model
                    if ema_enabled and ema_policy_model is not None:
                        ema_save_path = os.path.join(save_path, f"ema")

                        # Save EMA model - WAN feature extractor automatically excluded via custom state_dict()
                        os.makedirs(ema_save_path, exist_ok=True)

                        # The custom state_dict() method automatically handles the exclusion
                        ema_state_dict = ema_policy_model.state_dict()
                        ema_model_path = os.path.join(ema_save_path, "pytorch_model.bin")
                        torch.save(ema_state_dict, ema_model_path)
                        logger.info(f"✅ Saved EMA model (WAN feature extractor automatically excluded) to {ema_model_path}")

                    logger.info(f"Saved state to {save_path}")

                    # Upload checkpoint to wandb as artifact (only on main process)
                    if accelerator.is_main_process and args.report_to == "wandb" and is_wandb_available():
                        try:
                            artifact = wandb.Artifact(
                                name=f"checkpoint-{global_step}",
                                type="model",
                                description=f"Model checkpoint at step {global_step}",
                                metadata={
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "learning_rate": lr_scheduler.get_last_lr()[0],
                                }
                            )

                            # Add main model checkpoint
                            if os.path.exists(model_checkpoint_path):
                                artifact.add_file(model_checkpoint_path, name=f"checkpoint-{global_step}/pytorch_model.pt")

                            # Add EMA model if exists
                            if ema_enabled and ema_policy_model is not None:
                                if os.path.exists(ema_model_path):
                                    artifact.add_file(ema_model_path, name=f"checkpoint-{global_step}/ema/pytorch_model.bin")

                            # Log artifact to wandb
                            wandb.log_artifact(artifact)
                            logger.info(f"📤 Uploaded checkpoint-{global_step} to wandb as artifact")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to upload checkpoint to wandb: {e}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        policy_model,    # Use main model for sampling (EMA model used for final inference)
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            logs = {"loss": loss.detach().item(), 
                   "lr": lr_scheduler.get_last_lr()[0],
                   "token_compressor_features": args.use_realtime_framepack_features}
            
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            # logger.info(logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save model with filtered state dict to exclude WAN feature extractor
        unwrapped_model = accelerator.unwrap_model(policy_model)
        filtered_state_dict = unwrapped_model.state_dict()
        final_model_path = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(filtered_state_dict, final_model_path)
        logger.info(f"✅ Saved final filtered model (WAN feature extractor excluded) to {final_model_path}")
        
        # Final EMA model save
        if ema_enabled and ema_policy_model is not None:
            ema_save_path = os.path.join(args.output_dir, f"ema")

            # Save EMA model - WAN feature extractor automatically excluded via custom state_dict()
            os.makedirs(ema_save_path, exist_ok=True)

            # The custom state_dict() method automatically handles the exclusion
            ema_state_dict = ema_policy_model.state_dict()
            ema_model_path = os.path.join(ema_save_path, "pytorch_model.bin")
            torch.save(ema_state_dict, ema_model_path)
            logger.info(f"✅ Saved final EMA model (WAN feature extractor automatically excluded) to {ema_model_path}")
            logger.info(f"💡 Use EMA model for inference to get better performance")
        elif not ema_enabled:
            logger.info("⚠️ EMA is disabled - no EMA model saved")
        else:
            logger.info("No EMA model to save (non-main process or EMA disabled)")

        logger.info(f"Saved Model to {args.output_dir}")

        # Upload final model to wandb as artifact
        if args.report_to == "wandb" and is_wandb_available():
            try:
                final_artifact = wandb.Artifact(
                    name="final-model",
                    type="model",
                    description="Final trained model",
                    metadata={
                        "global_step": global_step,
                        "total_epochs": args.num_train_epochs,
                        "final_learning_rate": lr_scheduler.get_last_lr()[0],
                        "ema_enabled": ema_enabled,
                    }
                )

                # Add main model
                if os.path.exists(final_model_path):
                    final_artifact.add_file(final_model_path, name="pytorch_model.bin")

                # Add EMA model if exists
                if ema_enabled and os.path.exists(ema_model_path):
                    final_artifact.add_file(ema_model_path, name="ema/pytorch_model.bin")

                # Add config files
                if os.path.exists(args.config_path):
                    final_artifact.add_file(args.config_path, name="config.yaml")
                if os.path.exists(args.model_config_path):
                    final_artifact.add_file(args.model_config_path, name="model_config.yaml")

                # Log final artifact
                wandb.log_artifact(final_artifact)
                logger.info(f"📤 Uploaded final model to wandb as artifact")
                logger.info(f"   Main model: {final_model_path}")
                if ema_enabled:
                    logger.info(f"   EMA model: {ema_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to upload final model to wandb: {e}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
                # ignore_patterns=["step_*", "epoch_*"],
            )

    # Clean up the integrated framepack extractor if used
    if args.use_realtime_framepack_features and hasattr(policy_model, 'cleanup'):
        policy_model.cleanup()
            
    accelerator.end_training()
