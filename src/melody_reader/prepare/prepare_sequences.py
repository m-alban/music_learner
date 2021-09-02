import src.utils as utils

import json
import os
import pickle

from typing import List

SEQUENCE_DATA_PATH = os.path.join(
    utils.PROJECT_ROOT,
    'src',
    'melody_reader',
    'prepare',
    'data',
    'sequence_data.json'
)


def load_alphabet() -> List[str]:
    """ Load the semantic representation alphabet, first saving if necessary.
    """
    if not os.path.exists(SEQUENCE_DATA_PATH):
        save_alphabet()
    with open(SEQUENCE_DATA_PATH, 'r') as f:
        sequence_data = json.load(f)
    return sequence_data['alphabet']
        
def max_sequence_length() -> int:
    """ Get the max sequnce length of semantic representations, first saving if necessary. 
    """
    if not os.path.exists(SEQUENCE_DATA_PATH):
        save_alphabet()
    with open(SEQUENCE_DATA_PATH, 'r') as f:
        sequence_data = json.load(f)
    return sequence_data['max_sequence']

def save_alphabet():
    """ Generate the alphabet of the semantic representations of the Primus
    dataset.

    The dataset will be saved in 
    <project root>/src/melody_reader/prepare/data/sequence_data.json,
    with pairs 'alphabet':alphabet, 'max_sequence':max_sequence_length.
    """
    print('Preparing primus alphabet...')
    configs = utils.load_configs()
    primus_path = configs['primus_dataset_path']
    alphabet = set([])
    # max sequence length needed for crnn
    max_sequence = 0
    packages = ['package_aa', 'package_ab']
    packages = [os.path.join(primus_path, p) for p in packages]
    for subdir in packages:
        print('    Processing package:')
        print(f'        {subdir}')
        samples = os.listdir(subdir)
        for sample in samples:
            # each sample has a directory with files having the name of the directory
            semantic_file = sample + '.semantic'
            file_path = os.path.join(subdir, sample, semantic_file)
            with open(file_path, 'r') as f:
                # each semantic file has one line for the incipit
                line = f.readlines()[0]
            # ignore empty string
            tokens = [t for t in line.split('\t') if t]
            if len(tokens) > max_sequence:
                max_sequence = len(tokens)
            alphabet.update(set(tokens))
    sequence_data = {
        'alphabet': list(alphabet),
        'max_sequence': max_sequence
    }
    with open(SEQUENCE_DATA_PATH, 'w') as f:
        json.dump(sequence_data, f, indent = 4)
