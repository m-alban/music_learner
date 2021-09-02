import src.utils as utils

import random
import tensorflow as tf

from tensorflow.python.framework.ops import EagerTensor

def load_image(image_path: str) -> EagerTensor:
    image = tf.keras.preprocessing.image.load_img(image_path)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    height = image_array.shape[0]
    width = image_array.shape[1]
    # resize to height of 128 per paper
    #TODO: magic constant...
    resize_ratio = 128/height
    new_width = int(resize_ratio*width)
    tf.image.resize(image, [128, new_width], preserve_aspect_ratio = True)

class DataLoader():
    """ A class for loading and preparing data.

    Attributes:
        train_set: List[str], paths to train sample directories.
        test_set: List[str], paths to test sample directories. 
        val_set: List[str], paths to validation sample directories. 
    """

    def __init__(
        self,
        image_height: int,
        train_proportion: float,
        seed: int
    )
    """
    Args:
        image_height: The height to which the image will be resized
        train_proportion: the proportion of the primus dataset for training.
            The rest is split for test/val.
        seed:

    """
    self.image_height = image_height
    configs = utils.load_configs()
    primus_path = configs['primus_dataset_path']
    sample_paths = []
    # get paths to lowest directories
    for root, dirs, files in os.walk(primus_path):
        if not dirs:
            sample_paths.append(root)
    sample_paths = random.Random(seed).shuffle(sample_paths)
    train_index = int(len(sample_paths) * train_proportion)
    self.train_set = sample_paths[:train_index]
    holdout_set = sample_paths[train_index:]
    test_index = int(len(holdout_set)/2)
    self.test_set = holdout_set[:test_index]
    self.val_set = holdout_set[test_index:]

    @tf.function
    def read_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels = 1, dtype = tf.float32)
        return image

    @tf.function
    def resize(self, image):
        height = image_array.shape[0]
        width = image_array.shape[1]
        # resize to height of 128 per paper
        #TODO: magic constant...
        resize_ratio = self.image_height/height
        new_width = int(resize_ratio*width)
        tf.image.resize(image, [self.image_height, new_width], preserve_aspect_ratio = True)

    @tf.function
    def normalize(self, image):
        pixel_min = tf.reduce_min(image)
        pixel_range = tf.reduce_max(image) - pixel_min
        image = (image - pixel_min)/pixel_range
        return image

    @tf.function
    def preprocess(self, image_path):
        image = self.read_image(image_path)
        image = self.resize(image)
        image = self.normalize(image)
        return image

    def load_train(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.train_set)
        dataset = dataset.map(
            self.preprocess, 
            num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size = 128)
