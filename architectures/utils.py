"""
This file contains utility functions for the distillation training.
"""
import argparse
from fairseq.models.lightconv import base_architecture
from torch.nn import Embedding
import os.path as P
import os
from torch import load as load_tensor

def load_embs(emb_path):
    """
    Loads the Labse embeddings from the given path.

    emb_path - str
    """
    return Embedding.from_pretrained(load_tensor(emb_path, weights_only=True), padding_idx=0)

def create_folder(path):
    """
    Creates a folder at the given path if it does not exist.

    path - str
    """
    if not P.exists(path):
        os.mkdir(path)