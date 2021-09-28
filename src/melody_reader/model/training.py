import src.utils as utils
import src.melody_reader.prepare as prepare
from src.melody_reader.model import CRNN
from src.melody_reader.metrics import SequenceAccuracy

import os
import tensorflow as tf
from tensorflow.keras import callbacks

def train_model():
    configs = utils.Configs('melody_reader')
    loader_kwargs = configs.loader
    loader = prepare.DataLoader(**loader_kwargs)
    alphabet_size = loader.word_lookup.size()
    model = CRNN(alphabet_size)
    train_configs = configs.trainer
    optim = tf.keras.optimizers.Adam(learning_rate = train_configs['learning_rate'])
    # TODO: run_eagerly, otherwise iterator tensors in model.fit
    model.compile(
        optimizer = optim,
        run_eagerly = True,
        metrics = SequenceAccuracy()
    )
    #load data
    train_data = loader.load_partition('train')
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = loader.load_partition('val')
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    # model train setup
    checkpoint_path = ['src', 'melody_reader', 'model', 'saved_models']
    checkpoint_dir = os.path.join(utils.PROJECT_ROOT, *checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_name = 'model-{epoch:02d}-{val_sequence_accuracy:.2f}.ckpt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint_callback = callbacks.ModelCheckpoint(
        checkpoint_path, monitor = 'val_sequence_accuracy',
        verbose=1, save_best_only = False, save_weights_only = True,
        save_frequency = 1
    )
    model.fit(
        train_data,
        validation_data=val_data, 
        epochs=train_configs['epochs'], 
        callbacks = checkpoint_callback)
    
