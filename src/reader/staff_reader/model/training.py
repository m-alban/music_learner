import src.utils as utils
from src.reader.staff_reader import prepare, model, metrics

import os
import tensorflow as tf
from tensorflow.keras import callbacks

def train_model():
    configs = utils.Configs('staff_reader')
    loader_kwargs = configs.loader
    loader = prepare.DataLoader(**loader_kwargs)
    alphabet_size = loader.word_lookup.size()
    crnn_model = model.CRNN(alphabet_size)
    train_configs = configs.trainer
    optim = tf.keras.optimizers.Adam(learning_rate = train_configs['learning_rate'])
    # TODO: run_eagerly, otherwise iterator tensors in crnn_model.fit
    crnn_model.compile(
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
    checkpoint_path = ['src', 'reader', 'staff_reader', 'model', 'saved_models']
    checkpoint_dir = os.path.join(utils.project_root(), *checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_name = 'model-{epoch:02d}-{val_sequence_accuracy:.2f}.ckpt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint_callback = callbacks.ModelCheckpoint(
        checkpoint_path, monitor = 'val_sequence_accuracy',
        verbose=1, save_best_only = False, save_weights_only = True,
        save_frequency = 1
    )
    crnn_model.fit(
        train_data,
        validation_data=val_data, 
        epochs=train_configs['epochs'], 
        callbacks = checkpoint_callback)
    
