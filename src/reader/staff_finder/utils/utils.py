import src.utils as src_utils
import src.reader.staff_finder.model as staff_model

import os
import re
from PIL import Image
import torch

from typing import Tuple, List

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
