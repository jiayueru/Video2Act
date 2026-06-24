# import packages and module here
import sys, os
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _find_model_config(usr_args):
    """Resolve the training model_config used to locate fixed language embeddings."""
    explicit = usr_args.get("model_config_path")
    if explicit:
        return explicit

    ckpt_setting = usr_args.get("ckpt_setting")
    task_name = usr_args.get("task_name")
    names = []
    if ckpt_setting:
        names.append(os.path.basename(str(ckpt_setting).rstrip(os.sep)))
    if task_name:
        names.extend([f"{task_name}_wan", task_name])

    seen = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        candidate = os.path.join(parent_directory, "model_config", f"{name}.yml")
        if os.path.isfile(candidate):
            print(f"Using model_config_path for eval: {candidate}")
            return candidate
    return None


def _resolve_pretrained_path(ckpt_setting, checkpoint_id):
    ckpt_setting = str(ckpt_setting)
    if os.path.isfile(ckpt_setting):
        return ckpt_setting

    if os.path.isdir(ckpt_setting):
        candidates = [
            os.path.join(ckpt_setting, "pytorch_model", "mp_rank_00_model_states.pt"),
            os.path.join(ckpt_setting, "mp_rank_00_model_states.pt"),
            os.path.join(ckpt_setting, "pytorch_model.bin"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    if os.path.isabs(ckpt_setting) or "checkpoint-" in ckpt_setting:
        return os.path.join(ckpt_setting, "pytorch_model", "mp_rank_00_model_states.pt")

    return os.path.join(
        parent_directory,
        f"checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model/mp_rank_00_model_states.pt",
    )


def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


def get_model(usr_args):  # keep
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_id = usr_args.get("checkpoint_id")
    left_arm_dim, right_arm_dim, action_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["action_step"],
    )
    pretrained_path = _resolve_pretrained_path(ckpt_setting, checkpoint_id)
    print(f"Using pretrained checkpoint for eval: {pretrained_path}")
    model = Video2Act(
        pretrained_path,
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        action_step,
        config_path=usr_args.get("config_path"),
        model_config_path=_find_model_config(usr_args),
        fixed_lang_embed=usr_args.get(
            "fixed_lang_embed",
            usr_args.get(
                "fix_lang_embed",
                usr_args.get(
                    "fix_lang_embedded",
                    usr_args.get("fix_lang_embeded", usr_args.get("fixed_lang_embedded", False)),
                ),
            ),
        ),
        fixed_lang_embed_path=usr_args.get("fixed_lang_embed_path"),
        fixed_wan_text_embed_path=usr_args.get("fixed_wan_text_embed_path"),
    )
    return model


def eval(TASK_ENV, model, observation):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_eval_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[:model.action_step, :]  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
