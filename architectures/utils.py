"""
This file contains utility functions for the distillation training.
"""
import argparse
from fairseq.models.lightconv import base_architecture
from torch.nn import Embedding
import os.path as P
import os
from torch import load as load_tensor

def get_args(layers, kernel_sizes = [31], conv_type="lightweight", weight_softmax=True):
    """
    Sets the arguments for the LightConv model.

    layers - int
    kernel_sizes - List[int] (default - all 31)
    conv_type - (lightweight|dynamic)
    weight_softmax - bool
    """
    args = argparse.Namespace()
    args.encoder_layers = layers
    args.encoder_kernel_size_list = kernel_sizes

    args.encoder_conv_type = conv_type
    args.weight_softmax = weight_softmax

    args.encoder_embed_dim = 768
    args.max_source_positions = 1024
    base_architecture(args)
    return args

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