import src.utils as utils
from src.melody_reader import prepare

import os
import random
import tensorflow as tf

from tensorflow.python.data.ops.dataset_ops import ParallelMapDataset

class DataLoader:
    """ A class for loading and preparing data.

    Attributes:
        image_height: tensorflow.constant, new image height
        test_set: List[str], paths to test sample directories. 
        train_set: List[str], paths to train sample directories.
        val_set: List[str], paths to validation sample directories. 
        word_index: tf.lookup.StaticHashTable, maps a word to its index
            in the semantic alphabet.
        word_lookup: tf.lookup.StaticHashTable, maps an index 
            to its corresponding word.
    """

    def __init__(
        self,
        image_height: int,
        train_proportion: float,
        seed: int) -> None:
        """
        Args:
            image_height: The height to which the image will be resized
            train_proportion: the proportion of the primus dataset for training.
                The rest is split for test/val.
            seed: For reproducibility of train/test/val split.
        """
        # semantic word to integer mapping
        alphabet = prepare.load_alphabet()
        word_index = {}
        for idx, word in enumerate(alphabet):
            word_index[word] = idx+1
        index_vector = tf.constant(list(word_index.values()), dtype=tf.int32)
        word_vector = tf.constant(list(word_index.keys()))
        word_table_init = tf.lookup.KeyValueTensorInitializer(
            keys = word_vector,
            values = index_vector
        )
        self.word_index = tf.lookup.StaticHashTable(
            word_table_init,
            default_value = 0
        )
        word_lookup_init = tf.lookup.KeyValueTensorInitializer(
            keys = index_vector,
            values = word_vector
        )
        self.word_lookup = tf.lookup.StaticHashTable(
            word_lookup_init,
            default_value = ''
        )
        self.image_height = tf.constant(image_height)
        configs = utils.load_configs()
        # Load sample paths
        primus_path = configs['primus_dataset_path']
        sample_paths = prepare.load_sample_paths()
        sample_paths = [os.path.join(primus_path, s) for s in sample_paths]
        # Shuffle and partition
        random.Random(seed).shuffle(sample_paths)
        train_index = int(len(sample_paths) * train_proportion)
        self.train_set = sample_paths[:train_index]
        holdout_set = sample_paths[train_index:]
        test_index = int(len(holdout_set)/2)
        self.test_set = holdout_set[:test_index]
        self.val_set = holdout_set[test_index:]

    def load_partition(self, partition: str) -> ParallelMapDataset:
        """ Load a partition of the data, applying all preprocessing.

        Args:
            partiton: 'train', 'test', 'val'.
        
        Returns:
            A tensorflow mapped dataset.

        Raises:
            ValueError: If the partition argument is not in
                {'train', 'test', 'val'}
        """
        if partition == 'train':
            samples = self.test_set
        elif partition == 'test':
            samples = self.test_set
        elif partition == 'val':
            samples = self.val_set
        else:
            raise ValueError("Load partition should be 'train', 'test', 'val'")
        #TODO: remember to remove slicing
        images = [self.image_path(s) for s in samples]
        sequences = [self.sequence_path(s) for s in samples]
        dataset = tf.data.Dataset.from_tensor_slices((images, sequences))
        dataset = dataset.map(
            self.sample_preprocess) 
        # TODO: the below line leads to core dump errors
        #    num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size = 8, 
            drop_remainder=False,
            padded_shapes = (
                [tf.get_static_value(self.image_height), None, 1],
                [None], 
                []
            ),
            padding_values = (0., -1, 0)
        )
        return dataset

    @tf.function
    def image_normalize(self, image: tf.Tensor) -> tf.Tensor:
        """ Normalize pixels in the image."""
        pixel_min = tf.reduce_min(image)
        pixel_range = tf.reduce_max(image) - pixel_min
        image = (image - pixel_min)/pixel_range
        return image

    def image_path(self, sample_path: str) -> str:
        """Convert a sample path to a path to the corresponding image."""
        image_name = os.path.basename(sample_path) + '.png'
        return os.path.join(sample_path, image_name)

    @tf.function
    def image_preprocess(self, image_path: str) -> tf.Tensor:
        """Load and preprocess image."""
        image = self.image_read(image_path)
        image = self.image_resize(image)
        image = self.image_normalize(image)
        return image

    @tf.function
    def image_read(self, image_path: str) -> tf.Tensor:
        """Read the image at the given path."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels = 1, dtype = tf.uint8)
        return image

    @tf.function
    def image_resize(self, image: tf.Tensor) -> tf.Tensor:
        """Resize image to fixed height with proportional width."""
        # TODO slicing leads to autograph issues, with gast 0.4.0
        # suggested downgrade to 0.3.3, not done yet
        height = tf.shape(image)[0]
        #cast for multiplication with ratio
        width = tf.cast(tf.shape(image)[1], tf.float64)
        resize_ratio = self.image_height/height
        new_width = tf.cast(resize_ratio*width, tf.int32)
        resized = tf.image.resize(
            image, 
            [self.image_height, new_width], 
            preserve_aspect_ratio = True
        )
        return resized
   
    @tf.function
    def sample_preprocess(self, image_path, sequence_path):
        """
        """
        image = self.image_preprocess(image_path)
        sequence, sequence_length = self.sequence_load(sequence_path)
        return image, sequence, sequence_length

    @tf.function
    def sequence_load(self, sequence_path: str) -> tf.Tensor:
        """Load the label sequence"""
        sequence = tf.io.read_file(sequence_path)
        sequence = tf.strings.split(sequence)
        #TODO This is the line that causes issues with AUTOTUNE parallel calls
        sequence = tf.vectorized_map(lambda t: self.word_index[t], sequence)
        sequence_length = tf.size(sequence, out_type=tf.dtypes.int32)
        return sequence, sequence_length

    def sequence_path(self, sample_path: str) -> str:
        """Convert a sample path to a path to the corresponding 
        semantic representation.
        """
        sequence_name = os.path.basename(sample_path) + '.semantic'
        return os.path.join(sample_path, sequence_name)
