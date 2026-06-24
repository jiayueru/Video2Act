import sys
from pathlib import Path


VIDEO2ACT_ROOT = Path(__file__).resolve().parents[1]
if str(VIDEO2ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO2ACT_ROOT))

from config_utils import load_yaml_with_env


def read_yaml_value(file_path, key):
    data = load_yaml_with_env(file_path)
    value = data.get(key)
    if value is not None:
        print(value)
    else:
        print(f"Key '{key}' not found in {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python read_yaml.py <file_path> <key>")
        sys.exit(1)

    file_path = sys.argv[1]
    key = sys.argv[2]
    read_yaml_value(file_path, key)
