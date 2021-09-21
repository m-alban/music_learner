from src.melody_reader.model import CRNN
from src.melody_reader.prepare import DataLoader

import tensorflow as tf

def train_primus():
    """
    """
    loader = DataLoader(128, 0.7, 42)

    model = CRNN()
