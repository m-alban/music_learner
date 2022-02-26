import PIL
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from typing import Dict, Optional, Tuple, Union

class SampleCompose:
    """Composes transformations on image-target pair.

    From https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self, 
        image: Union[torch.Tensor, PIL.Image.Image],
        target:Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

#TODO: not used at the moment, using albumentations
class SampleRandomResizedCrop(T.RandomResizedCrop):
    """Performs RandomResizedCrop on an image-target pair 
    """
    def forward(
        self, 
        image: torch.Tensor, 
        target:Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        


class SampleToTensor(torch.nn.Module):
    """Converts an image-target pair with a PIL image to an image-target pair with a Tensor image.
    
    From https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
    """
    def forward(
        self, 
        image: PIL.Image.Image, 
        target:Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target
