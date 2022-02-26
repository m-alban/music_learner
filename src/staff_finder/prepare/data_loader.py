import src.utils as utils
from src.staff_finder.prepare.transforms import SampleCompose, SampleToTensor, SampleRandomResizedCrop

import albumentations as A
import collections
import glob
import io
import numpy as np
import os
import pandas as pd
import pathlib
from PIL import Image
import pytorch_lightning as pl
import random
import torch
import xml.etree.ElementTree as ET

from typing import List, Optional, Dict, Tuple

class MuscimaDataLightning(pl.LightningDataModule):
    """Lightning data module for training the staff finder model.

    The data module loads data from the path given in the staff finder 
        section of the configs. The proportion of train data is passed 
        in the constructor or loaded from configs.
        The remaining data is split in half for test and validation data.
        Data is transformed under one of
        1) Gaussian blur, 2) glass blur, or 3) motioin blur, followed
        by random scaling.
    """
    def __init__(self, train_proportion: float = None) -> None:
        """
        Args: 
            train_proportion: the proportion of data that will be used for the train dataset.
        """
        super().__init__()
        configs = utils.Configs('staff_finder')
        data_path = configs.data_path
        if not train_proportion:
            train_proportion = configs.loader['train_proportion']
        self.train_proportion = train_proportion
        annotation_dir = os.path.join(data_path, 'annotations')
        self.annotation_files = glob.glob(os.path.join(annotation_dir, '*.xml'))
        self.transforms = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit = (3, 11)),
                A.GlassBlur(),
                A.MotionBlur(blur_limit = (13, 15)),
            ], p = 0.5),
            #A.GaussNoise(10., 25.),
            A.RandomScale(scale_limit=0.2),
            A.pytorch.ToTensorV2(p=1.0)
        ])
        self.batch_size = configs.loader['batch_size']
        self.collate_fn = lambda batch: tuple(zip(*batch))
    
    def create_loader(self, 
        annotation_files: List[str], 
        transforms, 
        **loader_kwargs) -> torch.utils.data.DataLoader:
        """Creates a DataLoader of MuscimaDataset for annotations.

        Args:
            annotation_files: The annotation files for the samples to be loaded.
            transforms: the transforms to be applied to the images.
            loader_kwargs: extra arguments for a pytorch DataLoader
        Returns:
            The constructed DataLoader.
        """
        dataset = MuscimaDataset(annotation_files, transforms)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = self.batch_size,
            num_workers = os.cpu_count() - 1,
            collate_fn = self.collate_fn,
            **loader_kwargs
        )
        return loader

    def setup(self, stage: Optional[str] = None) -> None:
        """Splits self.annotation_files into train/test/validation sets.
        """
        random.Random(42).shuffle(self.annotation_files)
        train_count = int(len(self.annotation_files)*self.train_proportion)
        holdout_count = len(self.annotation_files) - train_count
        test_count = int(holdout_count/2)
        self.train_annotation_files = self.annotation_files[:train_count]
        holdout_annotation_files = self.annotation_files[train_count:]
        self.test_annotation_files = holdout_annotation_files[:test_count]
        self.val_annotation_files = holdout_annotation_files[test_count:]
   
    def train_dataloader(self):
        return self.create_loader(self.train_annotation_files, self.transforms, shuffle = True)

    def test_dataloader(self):
        return self.create_loader(self.test_annotation_files, SampleCompose([SampleToTensor()]))

    def val_dataloader(self):
        return self.create_loader(self.val_annotation_files, SampleCompose([SampleToTensor()]))
        

class MuscimaDataset(torch.utils.data.Dataset):
    """Class for loading muscima images and annotations
    """
    def __init__(self, annotation_files: List[str], transforms=None):
        """
        Args:
            annotation_files: the annotation files of the samples for the dataset.
            transforms: transformations to be applied to the images.
        """
        self.transforms=transforms
        configs = utils.Configs('staff_finder')
        data_path = configs.data_path
        self.image_dir = os.path.join(data_path, 'images')
        annotation_df = xml_to_csv(annotation_files)
        grouped_df = annotation_df.groupby('filename')
        data = collections.namedtuple('data', ['filename', 'dataframe'])
        self.image_groups = [
            data(filename, grouped_df.get_group(x))
            for filename, x in zip(grouped_df.groups.keys(), grouped_df.groups)
        ]

    def __getitem__(self, idx: int):
        group = self.image_groups[idx]
        image_path_list = xml_name_to_image_path(group.filename)
        image_path = os.path.join(self.image_dir, *image_path_list)
        img = Image.open(image_path).convert('RGB')
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for _, row in group.dataframe.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = xmin + row['width']
            ymax = ymin + row['height']
            area = row['height']*row['width']
            areas.append(area)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_label(row['class']))
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            if isinstance(self.transforms, A.Compose):
                img = np.array(img)/255.
                img = self.transforms(image = img)
                img = torch.as_tensor(img['image'], dtype=torch.float32)
            else:
                img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.image_groups)

def class_to_label(class_name: str) -> int:
    """Return numeric label for class name.
    """
    #TODO: provide class lookup elsewhere. not needed now since one class.
    classes = {'staff': 1}
    if class_name in classes:
        return classes[class_name]
    else:
        raise KeyError('Not a valid class name for box.')

def xml_to_csv(annotation_files: List[str]) -> pd.DataFrame:
    """Iterates through all .xml files and extracts bounding box data into a single Pandas dataframe.

    Args:
        annotation_files: Full paths to annotations to be loaded into csv.

    Returns:
        Dataframe of rows corresponding to boxes around staves, with the following columns:
        'filename', 'width', 'height', 'class', 'xmin', 'xmax', 'ymin', 'ymax' 
    """
    xml_list = []
    for xml_file in annotation_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        file_name = root.attrib['document']
        nodes = root.findall('Node')
        staff_lines = [n for n in nodes if n.findall('ClassName')[0].text=='staffLine']
        #TODO check len(staff_lines)%5 == 0
        for i in range(len(staff_lines)):
            # need to get the 1st and 5th line of a staff
            line_index = i%5
            if line_index not in [0, 4]:
                continue
            line = staff_lines[i]
            ymin = int(line.findall('Top')[0].text)
            xmin = int(line.findall('Left')[0].text)
            width = int(line.findall('Width')[0].text)
            height = int( line.findall('Height')[0].text)
            if not line_index:
                next_box = {
                    'ymin': ymin,
                    'xmin': xmin,
                }
            else: # line_index == 4
                next_box['ymax'] = ymin+height
                next_box['xmax'] = xmin+width
                height = next_box['ymax'] - next_box['ymin']
                row = (
                    file_name,
                    width,
                    height,
                    'staff',
                    next_box['xmin'],
                    next_box['ymin'],
                    next_box['xmax'],
                    next_box['ymax']
                )
                xml_list.append(row)
    columns = ['filename', 'width', 'height', 'class',
               'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=columns)
    return xml_df

def xml_name_to_image_path(xml_name: str) -> List[str]:
    """Gets the relative path of the image corresponding to the given xml file.

    Args:
        The filename of the annotation.

    Returns:
        The path of the image relative to the images folder.
    """
    xml_name_parts = xml_name.split('_')
    image_number = xml_name_parts[2].split('-')[1]
    image_name = f'p0{image_number}.png'
    image_path_list = [xml_name_parts[1].lower(), 'image', image_name]
    return image_path_list
