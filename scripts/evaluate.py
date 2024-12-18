"""
Evaluate the model on BUCC and FLORES datasets.

Usage:
python scripts/evaluate.py --model_class <path_to_model_class> --model_path <path_to_model_folder> --params_file <path_to_params_file> --output_file <path_to_output_file>

--model_class can be for example 
    architectures.light_convolution.LightConvModel
    architectures.baseline.labse.LabseModel
    architectures.baseline.input_emb.InputEmbAverageModel
--model_path is the path to the folder containing model weights and parameters, saved by train.py (for baselines leave empty)
--params_file is the path to the json file containing evaluation parameters
    params file should contain the following fields:
        BUCC:
            input_folder: str, path to the folder containing BUCC data
            output_folder: str, path to the folder to save intermediate results (required)
            batch_size: int, batch size for evaluation
            use_gpu: bool, whether to use GPU for evaluation
            verbose: bool, whether to print evaluation progress
            save_emb: bool, whether to save embeddings to the output folder
            save_pairs: bool, whether to save sentence pairs to the output folder
        FLORES:
            input_folder: str, path to the folder containing FLORES data
            batch_size: int, batch size for evaluation
            use_gpu: bool, whether to use GPU for evaluation
            verbose: bool, whether to print evaluation progress
--output_file is the path to the output csv file, which will contain model parameters and evaluation results
"""
import argparse
import csv
import os
import json
from evaluation.BUCC_evaluator import BUCCEvaluator
from evaluation.FLORES_evaluator import FLORESEvaluator
from evaluation.TATOEBA_evaluator import TATOEBAEvaluator
import json
import os.path as P
from datetime import datetime

def load_model(model_class, model_path=None):
    if (model_path != None): 
        params = json.load(open(P.join(model_path, "params.json"), "r"))
    else:
        params = {}
    
    model = model_class(**params)
    model.load_weights()
    return model, params

def run_evaluation(model, bucc_params, flores_params, tatoeba_params):

    evaluators = [FLORESEvaluator(**flores_params), TATOEBAEvaluator(**tatoeba_params), BUCCEvaluator(**bucc_params)]

    results = {}
    for evaluator in evaluators:
        now_results = evaluator.evaluate(model)
        for k, v in now_results.items():
            results[f"{evaluator.name}_{k}"] = v

    return results

def write_results_to_csv(data, output_file):
    # Load existing CSV data if file exists
    if os.path.isfile(output_file):
        existing_data = []
        with open(output_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            for row in reader:
                existing_data.append(row)
    else:
        existing_data = []
        fieldnames = []

    # Add new fieldnames from data that are not in existing fieldnames
    new_fieldnames = [field for field in data.keys() if field not in fieldnames]
    fieldnames = fieldnames + new_fieldnames

    # Write all data back to file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write existing rows
        for row in existing_data:
            # Fill in blank values for new columns
            row_data = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(row_data)
            
        # Write new row
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
    if model_params.get("name", None) == None:
        model_params["name"] = class_name

    # Run evaluations
    results = run_evaluation(model, eval_params["BUCC"], eval_params["FLORES"], eval_params["TATOEBA"])
    results["date"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Write results to CSV
    write_results_to_csv({**model_params, **results}, args.output_file)

    print(f"Evaluation completed. Results written to {args.output_file}")

if __name__ == "__main__":
    main()
