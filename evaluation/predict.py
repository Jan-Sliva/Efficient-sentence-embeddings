"""
This script is used to predict embeddings for a given set of sentences using a pre-trained model.

Usage:
python evaluation/predict.py --model_path <path_to_model_weights> --input_file <path_to_input_file> --output_file <path_to_output_file> --emb_path <path_to_file_with_word_emb_matrix>
"""
from architectures.light_conv_model import LightConvModel
from evaluation.custom_emb import CustomEmb

import argparse
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file for predictions.")
    parser.add_argument('--emb_path', type=str, required=True, help='Path to file with word embeddings matrix')

    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(f"Loading model from {args.model_path}")

    # Initialize the LightConvModel
    model = LightConvModel(args.emb_path)
    
    # Create CustomEmb instance
    custom_emb = CustomEmb(model, args.model_path, args.emb_dim)

    if args.verbose:
        print(f"Model loaded successfully")

    # Read sentences from input file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]

    # Predict embeddings
    embeddings = custom_emb.predict(sentences, args.batch_size, args.verbose)

    # Save predictions to output file
    np.save(args.output_file, embeddings)

    if args.verbose:
        print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()