import src.utils as utils

from object_detection.builders import model_builder
from object_detection.utils import config_util
import os
import tensorflow as tf

def load_model():
    proj_root = utils.PROJECT_ROOT
    model_dir = os.path.join(proj_root, 'src', 'staff_finder', 'model')
    config_path = os.path.join(model_dir, 'pipeline.config')
    checkpoint_dir = os.path.join(model_dir, 'checkpoint')
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    # restore from checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    manager = tf.compat.v2.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    return detection_model

#TODO: wrap in class to not pass model each time
@tf.function
def detect_fn(image, model):
    """Detect staves in image."""
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections

@tf.function
def image_read(image_path: str) -> tf.Tensor:
    """Load the image at the given path as uint8.

    Args:
        image_path: path to the .png file to be loaded.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels = 1, dtype = tf.uint8)
    return image


#def load_image
