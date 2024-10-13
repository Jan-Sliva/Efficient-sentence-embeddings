"""
Generate embeddings using a trained model.

Usage:
python scripts/predict.py --model_class <path_to_model_class> --model_path <path_to_model_folder> --input_file <path_to_input_file> --output_file <path_to_output_file> --batch_size <batch_size>

--model_class is the full path to the model class, for example:
    architectures.light_convolution.LightConvModel
    architectures.baseline.labse.LabseModel
    architectures.baseline.input_emb.InputEmbAverageModel
--model_path is the path to the folder containing model weights and parameters, saved by train.py (for baselines leave empty)
--input_file is the path to the input text file containing sentences
--output_file is the path to save the output embeddings
--batch_size is the batch size for prediction
"""
import argparse
import importlib
import json
import numpy as np
from typing import List
import pathlib as P

def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings using a trained model.')
    parser.add_argument("--model_class", type=str, required=True, help="Full path to the model class")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder with trained model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file containing sentences")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output embeddings")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for prediction")
    return parser.parse_args()

def load_model(model_class, model_path=None):
    if (model_path != None): 
        params = json.load(open(P.join(model_path, "params.json"), "r"))
    else:
        params = {}
    
    model = model_class(**params)
    model.load_weights()

    return model

def load_sentences(input_file: str) -> List[str]:
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def save_embeddings(embeddings: np.ndarray, output_file: str):
    np.save(output_file, embeddings)

def main():
    args = parse_args()

    # Dynamically import the model class
    module_name, class_name = args.model_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    # Load the model
    model, _ = load_model(ModelClass, args.model_path)

    # Load input sentences
    sentences = load_sentences(args.input_file)

    # Generate embeddings
    embeddings = model.predict(sentences, batch_size=args.batch_size, verbose=True)

    # Save embeddings
    save_embeddings(embeddings, args.output_file)

    print(f"Embeddings generated and saved to {args.output_file}")

if __name__ == "__main__":
    main()
