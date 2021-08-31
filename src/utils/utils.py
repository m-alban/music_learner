import json
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).absolute().parent.parent.parent

def load_configs():
    config_path = os.path.join(PROJECT_ROOT, 'configs.json')
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return configs
