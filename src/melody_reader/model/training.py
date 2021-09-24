import src.utils as utils
import src.melody_reader.prepare as prepare
from src.melody_reader.model import CRNN
from src.melody_reader.metrics import SequenceAccuracy

import tensorflow as tf

def train_model():
    configs = utils.Configs('melody_reader')
    loader_kwargs = configs.loader
    loader = prepare.DataLoader(**loader_kwargs)
    alphabet_size = loader.word_lookup.size()
    train_data = loader.load_partition('train')
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = loader.load_partition('val')
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    model = CRNN(alphabet_size)
    train_configs = configs.trainer
    optim = tf.keras.optimizers.Adam(learning_rate = train_configs['learning_rate'])
    # TODO: run_eagerly, otherwise iterator tensors in model.fit
    model.compile(
        optimizer = optim,
        run_eagerly = True,
        metrics = SequenceAccuracy()
    )
    steps_per_epoch = int(configs.num_samples*train_prop/batch_size)
    model.fit(
        train_data,
        validation_data=val_data, 
        epochs=train_configs['epochs'], 
        steps_per_epoch=steps_per_epoch
    )
    
