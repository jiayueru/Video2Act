"""Small config helpers shared by public Video2Act entry points."""

import os

import yaml


def expand_env_vars(value):
    """Recursively expand ${VAR} and $VAR strings loaded from YAML."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def load_yaml_with_env(path):
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return expand_env_vars(data)
