"""
This script saves the LaBSE embedding matrix to a file.

usage:
python save_labse_emb_matrix.py --path <path_to_file_with_labse_emb_matrix>
"""

import torch
from transformers import BertModel
import argparse

def main():
    parser = argparse.ArgumentParser(description='Save LaBSE embedding matrix to a file.')
    parser.add_argument('--path', type=str, required=True, help='Path to file where the embedding matrix will be saved.')
    args = parser.parse_args()

    path = args.path

    model= BertModel.from_pretrained("setu4993/LaBSE")

    torch.save(model.embeddings.word_embeddings.weight, path)

if __name__ == "__main__":
    main()