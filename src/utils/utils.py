import json
import os
import pathlib

from typing import Dict, Union

PROJECT_ROOT = pathlib.Path(__file__).absolute().parent.parent.parent

def dataset_path(component: str) -> str:
    """Load the dataset path for the project component.

    Args:
        component: Name of the project component.
    Returns
        Path to dataset.
    Raises:
        ValueError: if component is not in {'melody_reader', 'measure_finder'}.
    """
    configs = get_configs()
    return configs[component]['dataset_path']

def get_configs():
    """Load configs from json"""
    config_path = os.path.join(PROJECT_ROOT, 'configs.json')
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return configs

def loader_configs(component: str) -> Dict[str, Union[int, float]]:
    """Load the dataloader configs for the project component.

    Args:
        component: Name of the project component.
    Returns
        kwargs for the component's dataloader
    Raises:
        ValueError: if component is not in {'melody_reader', 'measure_finder'}.
    """
    configs = get_configs()
    return configs[component]['loader_configs']

def train_configs(component: str) -> Dict[str, Union[int, float]]:
    """Load the model training configs for the project component.

    Args:
        component: Name of the project component.
    Returns
        Configs for training.
    Raises:
        ValueError: if component is not in {'melody_reader', 'measure_finder'}.
    """
    configs = get_configs()
    return configs[component]['train_configs']


