"""
Train a model.

Usage:
python train.py --model_class <path_to_model_class> --params_file <path_to_params_json>

params file for LightConvModel should contain the following fields:
    paths:
        - emb_path: str, path to the embeddings file
        - data_folder: str, path to the folder containing training data
        - save_folder: str, path to save model checkpoints and parameters
        - tb_folder: str, path to save TensorBoard logs
    architecture:
        - layers: int, number of encoder layers
        - kernel_sizes: List[int], list of kernel sizes for each layer
        - conv_type: str, type of convolution ('lightweight' or 'dynamic')
        - weight_softmax: bool, whether to use softmax on weights
    training:
        - val_sentences: int, number of sentences to use for validation
        - lr: float, learning rate for the optimizer
        - batch_size: int, batch size for training
        - epochs: int, number of training epochs
        - percentage: float, percentage of training data to use in each epoch
        - report_each: int, number of epochs between each validationreport
"""
import argparse
import importlib
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run the distillation training.')
    parser.add_argument("--model_class", type=str, required=True, help="Full path to the model class")
    parser.add_argument("--params_file", type=str, required=True, help="Path to the JSON file containing model and evaluation parameters")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the parameters from the JSON file
    with open(args.params_file, 'r') as f:
        params = json.load(f)

    # Dynamically import the model class
    module_name, class_name = args.model_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)

    # Add a name to the model if it is not provided
    if params.get("name", None) is None:
        params["name"] = class_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Initialize the model
    model = ModelClass(**params)

    # Train the model
    model.train()

if __name__ == "__main__":
    main()
