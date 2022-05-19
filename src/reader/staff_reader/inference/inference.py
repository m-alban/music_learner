from src.reader import staff_reader
import src.utils

import os
import tensorflow as tf

def load_model():# -> CRNN:
    """Loads the latest checkpoint of the staff reader.

    Requires:
        checkpoints have been saved to 
            <PROJECT_ROOT>/src/reader/staff_reader/model/saved_models
    """
    # TODO: have to load the dataloader to get alphabet size. not ideal
    loader_kwargs = src.utils.Configs('staff_reader').loader
    loader = staff_reader.prepare.DataLoader(**loader_kwargs)
    alphabet_size = loader.word_lookup.size()
    model = staff_reader.model.CRNN(alphabet_size)
    checkpoint_dir = os.path.join(
        src.utils.project_root(), 'src', 'reader', 'staff_reader', 'model', 'saved_models'
    )
    latest_model = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest_model)
    return model
