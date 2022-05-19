from src.reader import staff_finder
import src.utils as src_utils

import os
import pathlib
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET

from typing import Dict, List, Tuple


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

def load_model(device: str = 'cpu') -> staff_finder.model.StaffFasterRCNN:
    """Loads the best staff finder model.
    
    Args:
        device: the device to load the model to.

    Raises:
        FileNotFoundError: if no checkpoints have been saved to 
            <project root>/src/staff_finder/model/checkpoint/
    """
    checkpoint_path = ['src', 'reader', 'staff_finder', 'model', 'checkpoint']
    checkpoint_dir = os.path.join(src_utils.project_root(), *checkpoint_path)
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

def load_boxes(xml_path: str) -> Dict[str, List[List[float]]]:
    """Loads object detection boxes to mapping of class name to boxes.

    Args:
        xml_path: path to an xml file with object detection boxes.
    Returns:
        keys: class names.
        values: lists of boxes in the format [xmin, ymin, xmax, ymax].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = {}
    nodes = root.findall('Node')
    for node in nodes:
        classname = node.find('ClassName').text
        xmin = float(node.find('xmin').text)
        ymin = float(node.find('ymin').text)
        xmax = float(node.find('xmax').text)
        ymax = float(node.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        if classname in boxes:
            boxes[classname].append(box)
        else:
            boxes[classname] = [box]
    return boxes

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
    tree = ET.ElementTree(root)
    for box, label in zip(boxes, labels):
        box_node = ET.Element('Node')
        classname_node = ET.Element('ClassName')
        classname_node.text = staff_finder.prepare.label_to_class[label]
        box_node.append(classname_node)
        coordinates = ['xmin', 'ymin', 'xmax', 'ymax']
        for coordinate_name, coordinate in zip(coordinates, box):
            node = ET.Element(coordinate_name)
            node.text = str(coordinate)
            box_node.append(node)
        root.append(box_node)
    return tree

def score_images(image_dir: str, output_dir: str) -> None:
    """Scores images at given filepaths and xml writes output.

        Scores will be written as xml files mirroring the structure of image_dir,
     with the base directory given by output_dir.

    Args:
        image_dir: base directory containing the images to be scored.
        output_dir: base directory to which xml files will be written.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    model.eval()
    dataset = StaffInferenceDataset(image_dir)
    configs = src_utils.Configs('staff_finder')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = configs.loader['batch_size'],
        collate_fn = lambda batch: tuple(zip(*batch)),
    )
    for img_batch, img_path_indices in loader:
        img_batch = [img.to(device) for img in img_batch]
        output = model(img_batch)
        output_trees = [output_to_xml(o) for o in output]
        del img_batch
        del output
        for i in range(len(output_trees)):
            tree = output_trees[i]
            image_path = dataset.index_to_path(img_path_indices[i])
            image_filename = os.path.basename(image_path)
            tree.getroot().set('filename', image_path)
            output_path = os.path.relpath(image_path, image_dir)
            xml_filename = image_filename.split('.')[0] + '.xml'
            output_path = os.path.join(output_dir, output_path)
            output_path = output_path.replace(image_filename, xml_filename)
            pathlib.Path(output_path).parents[0].mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                tree.write(f, encoding='utf-8')
