import src.utils as utils

import collections
import glob
import io
from object_detection.utils import dataset_util
import os
import pandas as pd
import pathlib
from PIL import Image
import random
import tensorflow.compat.v1 as tf
import xml.etree.ElementTree as ET

from typing import List

def create_tf_example(image_group, image_dir):
    """Creates a tf.train.Example for the group of boxes associated with the image

    Args:
        image_group: A named tuple whose fields are 'filename' and 'object', where tuple.object 
            is a pandas dataframe with columns filename, width, height, class, xmin, ymin, xmax, ymax
            indicating a box containing a staff.
    """
    # get image path from xml name
    xml_name = image_group.filename
    xml_name_parts = xml_name.split('_')
    image_number = xml_name_parts[2].split('-')[1]
    image_name = f'p0{image_number}.png'
    image_path_list = [xml_name_parts[1].lower(), 'image', image_name]
    image_path = os.path.join(image_dir, *image_path_list)
    # read image
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    # prepare data for tfrecord write
    filename = image_group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for i, row in image_group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(1) #TODO: only one class for this data, do we need a label?

    tf_features = tf.train.Features(
        feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    })
    tf_example = tf.train.Example(features=tf_features)
    return tf_example

def write_dataset():
    """ Writes TFRecords of train and test dataset for staff recognition.

    Uses 85/15 train test split
    """
    write_tf_images()
    # use resized images in the object detection folder
    configs = utils.Configs('staff_finder')
    data_path = configs.data_path
    annotations_dir = os.path.join(data_path, 'annotations')
    annotation_df = xml_to_csv(annotations_dir)
    # group annotations by image
    grouped_df = annotation_df.groupby('filename')
    data = collections.namedtuple('data', ['filename', 'object'])
    image_groups = [
            data(filename, grouped_df.get_group(x)) 
            for filename, x in zip(grouped_df.groups.keys(), grouped_df.groups)]
    random.Random(42).shuffle(image_groups)
    test_count = int(len(image_groups)*0.15)
    test_set = image_groups[:test_count]
    train_set = image_groups[test_count:]
    od_path = utils.object_detection_path()
    #TODO: setup requires copying in pretrained models so might not need the mkdir
    tf_annotations_path = os.path.join(od_path, 'workspace', 'staff_finder', 'annotations')
    if not os.path.isdir(tf_annotations_path):
        os.makedirs(tf_annotations_path)
    train_path = os.path.join(tf_annotations_path, 'train.record')
    test_path = os.path.join(tf_annotations_path, 'test.record')
    image_dir = os.path.join(od_path, 'workspace', 'staff_finder', 'images')
    write_records(train_set, train_path, image_dir)
    write_records(test_set, test_path, image_dir)

#TODO: for typing, would want a class inheriting from typing.NamedTuple
def write_records(group_list, out_path, image_dir):
    """ Write the tf examples in group_list to out_path

    Args:
        group_list: list with named tuples as elements, whose fields are 'filename' and 'object'
        out_path: str, the path to where the examples will be written
        image_dir: str, path to the directory containing images for the dataset
    """
    with tf.python_io.TFRecordWriter(out_path) as tf_writer:
        for group in group_list:
            tf_example = create_tf_example(group, image_dir)
            tf_writer.write(tf_example.SerializeToString())

def write_tf_images():
    """Resize images to the scale of the object detection model. 

    Write images to the images folder in <object_detection_path>/staff_finder/images
    """
    configs = utils.Configs('staff_finder')
    loader_configs = configs.loader
    image_out_size = (loader_configs['image_width'], loader_configs['image_height'])
    # setting up write destination
    od_path = utils.object_detection_path()
    #TODO: again, setup requires copying pretrained models so might not need the mkdir
    out_image_dir = os.path.join(od_path, 'workspace', 'staff_finder', 'images')
    # read original muscima data
    configs = utils.Configs('staff_finder')
    data_path = configs.data_path
    image_dir = os.path.join(data_path, 'images')
    glob_search = os.path.join(image_dir, '**', 'image', '*.png')
    image_paths = glob.glob(glob_search)
    for image_path in image_paths:
        # create same subdirectories 
        sub_path = image_path.split('images')[1]
        sub_path_components = sub_path.split(os.path.sep)
        out_path = os.path.join(out_image_dir, *sub_path_components)
        out_dir = os.path.dirname(out_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        image = Image.open(image_path)
        image = image.resize(image_out_size, resample = Image.BILINEAR)
        image.save(out_path)

def xml_to_csv(annotations_path: str):
    """Iterates through all .xml files and extracts bounding box data into a single Pandas dataframe.

    Args:
        annotations_path: <muscima_pp/v2.0>/annotations

    Returns:
        Dataframe with columns 'filename', 'xmin', 'xmax', 'ymax' where each row gives a box for a staff.
    """
    annotations = glob.glob(os.path.join(annotations_path, '*.xml'))
    xml_list = []
    for xml_file in annotations:
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

