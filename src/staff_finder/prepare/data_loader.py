import src.utils as utils

import os
import pathlib
import xml.etree.ElementTree as ET


from PIL import Image, ImageDraw

def prepare_image(image_path):
    """ <dataset_path>/images/w-01/image/p010.png
    """
    path_list = image_path.split(os.sep)
    image_number = int(path_list[-1][1:4])
    sample_group = path_list[-3].upper()
    #xml_path = os.path.join(*path_list[:-4], 'annotations')
    # TODO this is ugly...
    xml_path = pathlib.Path(image_path).parent.parent.parent.parent 
    xml_name = f'CVC-MUSCIMA_{sample_group}_N-{image_number}_D-ideal.xml'
    xml_path = os.path.join(xml_path, 'annotations', xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    nodes = root.findall('Node')
    staff_lines = [n for n in nodes if n.findall('ClassName')[0].text=='staffLine']
    
    #########################
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    line = staff_lines[0]
    top = int(line.findall('Top')[0].text)
    left = int(line.findall('Left')[0].text)
    width = int(line.findall('Width')[0].text)
    height = int( line.findall('Height')[0].text)
    line_draw = [(left, top), (left+width, top+height)]
    draw.line(line_draw, fill='yellow', width=10)
    image.show()

def main():
    configs = utils.Configs('staff_finder')
    data_path = utils.data_path
    image_dir = os.path.join(data_path, 'images')
    annotations_dir = os.path.join(data_path, 'annotations')


if __name__ == '__main__':
    main()

