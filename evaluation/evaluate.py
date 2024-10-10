"""
This script evaluates a model (Labse, WordEmb or custom pytorch model) on the BUCC 2018 and FLORES+ datasets.

usage:
python evaluation/evaluate.py --model name_of_model --model_path path/to/model/weights.pth --emb_path path/to/embeddings.pth --BUCC_folder path/to/BUCC/data --FLORES_folder path/to/FLORES/data --eval_folder path/to/eval/folder
python evaluation/evaluate.py --model labse --BUCC_folder path/to/BUCC/data --FLORES_folder path/to/FLORES/data --eval_folder path/to/eval/folder
python evaluation/evaluate.py --model word_emb --BUCC_folder path/to/BUCC/data --FLORES_folder path/to/FLORES/data --eval_folder path/to/eval/folder
"""
from evaluation.labse import Labse
from evaluation.word_emb import WordEmb
from evaluation.custom_emb import CustomEmb
from evaluation.evaluate_BUCC import BUCC_eval
from evaluation.evaluate_FLORES import FLORES_eval
from architectures.light_conv_model import LightConvModel

import os.path as P
import os
import sys
import torch

import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
    parser.add_argument("--use_gpu", type=bool, default=True)

    parser.add_argument("--model", type=str, help="\"labse\"|\"word_emb\"|\"light_conv\"")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--emb_path", type=str, default=None, help="Path to embeddings file for LightConvModel")

    parser.add_argument("--BUCC_folder", default=None, type=str)
    parser.add_argument("--FLORES_folder", default=None, type=str)
    parser.add_argument("--eval_folder", type=str)

    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()

    if not P.exists(args.eval_folder):
        os.mkdir(args.eval_folder)

    if args.model == "labse":
        model = Labse()
        pred_f = lambda x: model.predict(x, args.batch_size, args.verbose)
    elif args.model == "word_emb":
        model = WordEmb()
        pred_f = lambda x: model.predict(x, args.batch_size, args.verbose)
    elif args.model == "light_conv":
        if args.emb_path is None:
            raise ValueError("--emb_path must be provided for LightConvModel")
        light_conv_model = LightConvModel(args.emb_path)
        model = CustomEmb(light_conv_model, args.model_path, 768)
        pred_f = lambda x: model.predict(x, args.batch_size, args.verbose)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if args.FLORES_folder is not None:
        FLORES_eval(args.FLORES_folder, args.eval_folder, pred_f, args.model, args.use_gpu)

    if args.BUCC_folder is not None:
        BUCC_eval(args.BUCC_folder, args.eval_folder, pred_f, args.model, False, args.use_gpu)

if __name__ == "__main__":
    main()