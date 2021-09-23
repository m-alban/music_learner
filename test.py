import src.utils as utils
import src.melody_reader.prepare as reader_prep
from src.melody_reader.model import CRNN
from src.melody_reader.metrics import SequenceAccuracy

import tensorflow as tf

def data_prep():
    print('testing data prep')
    reader_prep.process_data()
    alphabet = reader_prep.load_alphabet()
    max_sequence_length = reader_prep.max_sequence_length()
    sample_paths = reader_prep.load_sample_paths()
    print(len(alphabet))
    print(max_sequence_length)
    print(len(sample_paths))

def test_data_loader():
    print('testing data loader')
    tf.executing_eagerly()
    loader_kwargs = utils.loader_configs('melody_reader')
    loader = reader_prep.DataLoader(**loader_kwargs)
    dataset = loader.load_partition('train')
    for image, seq, seq_len in dataset:
        pass
    #    print(image.shape)
    #    print(seq.shape)
    #    print(seq_len.shape)
    #    print(tf.reduce_max(seq_len))
    #    print(seq_len)
        #print(elem)

def test_model():
    print('testing model train')
    component = 'melody_reader'
    loader_kwargs = utils.loader_configs(component)
    loader = reader_prep.DataLoader(**loader_kwargs)
    alphabet_size = loader.word_lookup.size()
    train_data = loader.load_partition('train')
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = loader.load_partition('val')
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    model = CRNN(alphabet_size)
    # TODO: run_eagerly is on because otherwise getting iterator tensors
    # in model.fit
    train_configs = utils.train_configs(component)
    optim = tf.keras.optimizers.Adam(learning_rate=train_configs['learning_rate']),
    model.compile(
        optimizer = optim,
        run_eagerly = True,
        metrics = SequenceAccuracy()
    )
    model.fit(train_data,
              validation_data=val_data, 
              epochs=train_configs['epochs'], 
              steps_per_epoch=13152/8)
    #for image, sequence, seq_len in dataset:
    #    print('##############')
    #    out = model(image)
    #    print(out.shape[1])
    #    print('sequence_lengths: ', seq_len)
    #for batch in dataset:
    #    loss = model.train_step(batch)
    #    print('loss: ', loss)

def main():
    data_prep()
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    with tf.device('/gpu:0'):
        test_data_loader()
        test_model()

if __name__ == '__main__':
    main()
