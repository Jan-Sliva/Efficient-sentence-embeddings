import argparse
import csv
import os
import json
from evaluation.BUCC_evaluator import BUCCEvaluator
from evaluation.FLORES_evaluator import FLORESEvaluator
import json
import os.path as P

def load_model(model_class, model_path=None):
    if (model_path != None): 
        params = json.load(open(P.join(model_path, "params.json"), "r"))
    else:
        params = {}
    
    model = model_class(**params)
    model.load_weights()
    return model, params

def run_evaluation(model, bucc_params, flores_params):

    evaluators = [FLORESEvaluator(**flores_params), BUCCEvaluator(**bucc_params)]

    results = {}
    for evaluator in evaluators:
        now_results = evaluator.evaluate(model)
        for k, v in now_results.items():
            results[f"{evaluator.name}_{k}"] = v

    return results

def write_results_to_csv(data, output_file):
    # Get existing fieldnames if file exists, otherwise use all keys
    if os.path.isfile(output_file):
        with open(output_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            existing_fieldnames = next(reader, [])
    else:
        existing_fieldnames = []

    # Add new fieldnames from all_data that are not in existing_fieldnames
    new_fieldnames = [field for field in data.keys() if field not in existing_fieldnames]
    fieldnames = existing_fieldnames + new_fieldnames

    # Open file in append mode if it exists, otherwise write mode
    mode = 'a' if os.path.isfile(output_file) else 'w'
    
    with open(output_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if mode == 'w':
            writer.writeheader()
        
        # Fill in blank values for new columns
        row_data = {field: data.get(field, '') for field in fieldnames}
        writer.writerow(row_data)

def main():
    parser = argparse.ArgumentParser(description="Run BUCC and FLORES evaluations")
    parser.add_argument("--model_class", type=str, required=True, help="Full path to the model class")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
    parser.add_argument("--params_file", type=str, required=True, help="Path to the JSON file containing evaluation parameters")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    # Load parameters from JSON file
    with open(args.params_file, 'r') as f:
        eval_params = json.load(f)

    # Import the model class dynamically
    module_name, class_name = args.model_class.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # Initialize and load the model
    model, model_params = load_model(model_class, args.model_path)

    # Run evaluations
    results = run_evaluation(model, eval_params["BUCC"], eval_params["FLORES"])

    # Write results to CSV
    write_results_to_csv({**model_params, **results}, args.output_file)

    print(f"Evaluation completed. Results written to {args.output_file}")

if __name__ == "__main__":
    main()
