import src.melody_reader.prepare as reader_prep

import tensorflow as tf

def melody_reader():
    reader_prep.process_data()
    alphabet = reader_prep.load_alphabet()
    max_sequence_length = reader_prep.max_sequence_length()
    sample_paths = reader_prep.load_sample_paths()
    print(len(alphabet))
    print(max_sequence_length)
    print(len(sample_paths))

def test_data_loader():
    loader = reader_prep.DataLoader(128, 0.7, 42)
    train_images = loader.load_partition('train')
    #print(type(train_images))
    #s = set([])
    for image, seq, seq_len in train_images:
        print(image.shape)
        print(seq.shape)
        print(seq_len.shape)
        print(tf.reduce_max(seq_len))
        print(seq_len)
        #print(elem)
        #s.add(elem.shape[1])
    #print(len(s)) # 1559, use ragged tensors
    #print(sorted(s))

def main():
    #melody_reader()
    test_data_loader()

if __name__ == '__main__':
    main()
