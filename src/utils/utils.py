import json
import os
import pathlib

from typing import Dict, Union

PROJECT_ROOT = pathlib.Path(__file__).absolute().parent.parent.parent

def load_configs() -> Dict:
    config_path = os.path.join(PROJECT_ROOT, 'configs.json')
    with open(config_path, 'r') as f:
        configs = json.load(f)
    return configs

def object_detection_path():
    """TODO: maybe a better way...
    """
    return load_configs()['project_tools']['object_detection']['path']

class Configs:
    """Class for delivering component configurations.

    Attributes:
        data_path: str, path to the dataset
        loader: Dict[str, Union[int, float]], the dataloader kwargs
        trainer: Dict[str, Union[int, float]], trainer configs
    """
    def __init__(self, component: str) -> None:
        """Set up configs.

        Args:
            component: Name of the project component.
        Raises:
            ValueError: if component is not in 
                {'melody_reader', 'score_cleaner', 'staff_finder'}
        """
        if component not in ['melody_reader', 'score_cleaner', 'staff_finder']:
            raise ValueError
        configs = load_configs()
        component_configs = configs['project_components'][component]
        self.data_path = component_configs['dataset_path']
        self.loader = component_configs['loader_configs']
        self.trainer = component_configs['train_configs']
