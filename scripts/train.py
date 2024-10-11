import argparse
import importlib
import json

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

    # Initialize the model
    model = ModelClass(**params)

    # Train the model
    model.train()

if __name__ == "__main__":
    main()
