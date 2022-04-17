#from src.melody_reader.model import CRNN
from src import melody_reader
import src.utils

import os
import tensorflow as tf

def load_model():# -> CRNN:
    """Loads the latest checkpoint of the melody reader.

    Requires:
        checkpoints have been saved to 
            <PROJECT_ROOT>/src/melody_reader/model/saved_models
    """
    # TODO: have to load the dataloader to get alphabet size. not ideal
    loader_kwargs = src.utils.Configs('melody_reader').loader
    loader = melody_reader.prepare.DataLoader(**loader_kwargs)
    alphabet_size = loader.word_lookup.size()
    model = melody_reader.model.CRNN(alphabet_size)
    checkpoint_dir = os.path.join(
        src.utils.project_root(), 'src', 'melody_reader', 'model', 'saved_models'
    )
    latest_model = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest_model)
    return model
