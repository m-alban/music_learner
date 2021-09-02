import src.melody_reader.prepare as reader_prep

def melody_reader():
    alphabet = reader_prep.load_alphabet()
    max_sequence_length = reader_prep.max_sequence_length()
    print(len(alphabet))
    print(max_sequence_length)

def main():
    melody_reader()

if __name__ == '__main__':
    main()
