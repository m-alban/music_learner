import src.utils as src_utils
import src.staff_finder.model as staff_model

import os
import re

from typing import Tuple


def ckpt_sort_key(ckpt: str) -> Tuple[float, int]:
    """Returns a sort value for the given saved checkpoint file name.

    Arguments:
        ckpt: the checkpoint file name. 
    Raises:
        ValueError if ckpt is not named in the format 
            'staff-finder-epoch=EPOCH-val_map=VAL_MAP.ckpt'
    Returns:
        (mAP, epoch)
    """
    pattern = r'staff-finder-epoch=\d*-val_map=\d*\.?\d*\.ckpt'
    if not re.findall(pattern, ckpt):
        raise ValueError(
            'Model checkpoint file naming does not match standard, see'
            f'staff_finder.utils.ckpt_sort_key. Checkpoint name is "{ckpt}",'
            f'acceptable file name pattern is "{pattern}".'
        )
    epoch_match = re.search(r'epoch=\d*', ckpt).group(0)
    epoch = int(''.join([c for c in epoch_match if c.isdigit()]))
    map_match = re.search(r'val_map=\d*\.\d*', ckpt).group(0)
    map_val = float(''.join([c for c in map_match if c.isdigit() or c=='.']))
    return (map_val, epoch)

def load_model() -> staff_model.StaffFasterRCNN:
    """Loads the best staff finder model.

    Raises:
        FileNotFoundError: if no checkpoints have been saved to 
            <project root>/src/staff_finder/model/checkpoint/
    """
    checkpoint_path = ['src', 'staff_finder', 'model', 'checkpoint']
    checkpoint_dir = os.path.join(src_utils.PROJECT_ROOT, *checkpoint_path)
    checkpoint_list = os.listdir(checkpoint_dir)
    if not checkpoint_list:
        raise FileNotFoundError('Staff finder: No checkpoints saved.')
    checkpoint_list.sort(key = ckpt_sort_key, reverse = True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[0])
    model = staff_model.MuscimaLightning.load_from_checkpoint(checkpoint_path).model
    return model
