import src.utils as utils

import json
import os
import pickle

from typing import List

# used to save and load stored sequence data, e.g. alphabet
SEQUENCE_DATA_PATH = os.path.join(
    utils.PROJECT_ROOT,
    'src',
    'melody_reader',
    'prepare',
    'data',
    'sequence_data.json'
)

def load_alphabet() -> List[str]:
    """ Load the semantic representation alphabet, first saving if necessary."""
    if not os.path.exists(SEQUENCE_DATA_PATH):
        process_data()
    with open(SEQUENCE_DATA_PATH, 'r') as f:
        sequence_data = json.load(f)
    return sequence_data['alphabet']
        
def load_sample_paths() -> List[str]:
    """ Load the sample paths.

    Returns:
        A list of all samples of the format
        <package name><os separator><sample name>
    """
    if not os.path.exists(SEQUENCE_DATA_PATH):
        process_data()
    with open(SEQUENCE_DATA_PATH, 'r') as f:
        sequence_data = json.load(f)
    return sequence_data['sample_paths']

#TODO wont need this, sequence maxs are aggregates of each batch.
def max_sequence_length() -> int:
    """ Get the max sequnce length of semantic representations, first saving if necessary. 
    """
    if not os.path.exists(SEQUENCE_DATA_PATH):
        process_data()
    with open(SEQUENCE_DATA_PATH, 'r') as f:
        sequence_data = json.load(f)
    return sequence_data['max_sequence']

def process_data() -> None:
    """ Generate data for the Primus dataset to be used later.

    Save the sorted semantic representation alphabet, max sequence length,
    and sample directory names in the format 
    '<package><os separator><sample name>'.
    Data is saved at 
    <project root>/src/melody_reader/prepare/data/sequence_data.json,
    with keys 'alphabet', 'max_sequence', and 'sample_paths'
    """
    print('Preparing primus alphabet...')
    configs = utils.Configs('melody_reader')
    primus_path = configs.data_path
    alphabet = set([])
    # max sequence length needed for crnn
    max_sequence = 0
    # save sample paths for quick lookup later
    sample_paths = []
    for root, dirs, files in os.walk(primus_path):
        if dirs:
            # not yet at lowest diretory
            continue
        # at lowest directory, a sample directory
        semantic_file = [x for x in files if '.semantic' in x and '._' not in x][0]
        # keep package and sample name
        sample_name = os.path.join(*root.split(os.path.sep)[-2:])
        file_path = os.path.join(root, semantic_file)
        with open(file_path, 'r') as f:
            # each semantic file has one line for the incipit
            line = f.readline().strip()
        # ignore empty string
        tokens = line.split('\t') 
        if len(tokens) > max_sequence:
            max_sequence = len(tokens)
        alphabet.update(set(tokens))
        sample_paths.append(sample_name)
    sequence_data = {
        # sorting for consistent word indexing across project builds
        'alphabet': sorted(list(alphabet)),
        'max_sequence': max_sequence,
        'sample_paths': sample_paths
    }
    data_dir = os.path.dirname(SEQUENCE_DATA_PATH)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    with open(SEQUENCE_DATA_PATH, 'w') as f:
        json.dump(sequence_data, f, indent = 4)
