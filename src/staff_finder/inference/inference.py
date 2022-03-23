from src import staff_finder
import src.utils as src_utils

import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET

from typing import Dict, List, Tuple


def load_model(device: str = 'cpu') -> staff_finder.model.StaffFasterRCNN:
    """Loads the best staff finder model.
    
    Args:
        device: the device to load the model to.

    Raises:
        FileNotFoundError: if no checkpoints have been saved to 
            <project root>/src/staff_finder/model/checkpoint/
    """
    checkpoint_path = ['src', 'staff_finder', 'model', 'checkpoint']
    checkpoint_dir = os.path.join(src_utils.PROJECT_ROOT, *checkpoint_path)
    checkpoint_list = os.listdir(checkpoint_dir)
    if not checkpoint_list:
        raise FileNotFoundError('Staff finder: No checkpoints saved.')
    checkpoint_list.sort(key = staff_finder.utils.ckpt_sort_key, reverse = True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[0])
    model = (
        staff_finder.model.MuscimaLightning
        .load_from_checkpoint(checkpoint_path).model
    )
    return model.to(device)

class StaffInferenceDataset(torchvision.datasets.ImageFolder):
    """ImageFolder subclass that maintains image path when returning instances.
    """
    def __getitem__(self, index: int):
        image_path, _ = self.imgs[index]
        img = Image.open(image_path).convert('RGB')
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img)
        return img, index

    def index_to_path(self, index: int) -> str:
        """Gives image path given index
        """
        return self.imgs[index][0]

# TODO: write xml schema for transcription files
def output_to_xml(detection_out: Dict[str, torch.Tensor]) -> ET.ElementTree:
    """Converts output of object detection model to an xml object.

    Args:
        detection_out: the output of a detection model.
    Requires:
        detection_out has key-value pairs 
            'boxes': box corners in the format [xmin, ymin, xmax, ymax]
            'labels': box labels
            'scores': probabilities for predictions
    Returns:
        xml object detailing the boxes and classes predicted for the image.
    """
    boxes = detection_out['boxes'].tolist()
    labels = detection_out['labels'].tolist()
    root = ET.Element('Nodes')
    # TODO: decide what to set filename. maybe path, or dataset included
    root.set('filename', os.path.basename(image_path))
    tree = ET.ElementTree(root)
    for box, label in zip(boxes, labels):
        box_node = ET.ElementTree(root)
        elements = []
        classname_node = ET.Element('ClassName')
        classname_node.text = staff_finder.prepare.label_to_class[label]
        elements.append(classname_node)
        coordinates = ['xmin', 'ymin', 'xmax', 'ymax']
        for coordinate_name, coordinate in zip(coordinates, box):
            node = ET.Element(coordinate_name)
            node.text = str(coordinate)
            elements.append(node)
        box_node.extend(elements)
        root.append(box_node)
    return tree

def score_images(image_dir: str) -> None:
    """Scores images at given filepaths and writes output.

    scores will be written as xml files mirroring the structure of image_dir.

    Args:
        image_dir: base directory containing the images to be scored.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    model.eval()
    dataset = StaffInferenceDataset(image_dir)
    output_dir = os.path.join(src_utils.PROJECT_ROOT, 'data', 'transcriptions')
    configs = src_utils.Configs('staff_finder')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = configs.loader['batch_size'],
        collate_fn = lambda batch: tuple(zip(*batch))
    )
    for img_batch, img_path_indices in loader:
        img_batch = [img.to(device) for img in img_batch]
        output = model(img_batch)
        image_paths = [dataset.index_to_path(p) for p in img_path_indices]
        output_paths = [os.path.relpath(p, image_dir) for p in image_paths]
        output_paths = [os.path.join(output_dir, p) for p in output_paths]
