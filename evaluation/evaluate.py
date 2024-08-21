from evaluation.labse import Labse
from evaluation.word_emb import WordEmb
from evaluation.custom_emb import CustomEmb
from evaluation.evaluate_BUCC import BUCC_eval
from evaluation.evaluate_FLORES import FLORES_eval
from architectures.run_model import get_model

import os.path as P
import os
import sys
import torch

import argparse

def main():
  parser = argparse.ArgumentParser(description='Evaluation')
  parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
  parser.add_argument("--use_gpu", type=bool, default=True)

  parser.add_argument("--model", type=str, help="\"labse\"|\"word_emb\"|name_of_model")
  parser.add_argument("--model_path", type=str, default=None)

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
  else:
    model = CustomEmb(get_model(), args.model_path, 768)
    pred_f = lambda x: model.predict(x, args.batch_size, args.verbose)
    
  if args.FLORES_folder is not None:
    FLORES_eval(args.FLORES_folder, args.eval_folder, pred_f, args.model, args.use_gpu)

  if args.BUCC_folder is not None:
    BUCC_eval(args.BUCC_folder, args.eval_folder, pred_f, args.model, False, args.use_gpu)

  

if __name__ == "__main__":
  main()