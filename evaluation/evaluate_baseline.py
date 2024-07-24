from labse import Labse
from word_emb import WordEmb
from evaluate_BUCC import BUCC_eval
from evaluate_FLORES import FLORES_eval

import os.path as P
import os
import sys

import argparse

def main():
  parser = argparse.ArgumentParser(description='LaBse evaluation')
  parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
  parser.add_argument("--use_gpu", type=bool, default=True)

  parser.add_argument("--model", type=str, help="labse|word_emb")

  parser.add_argument("--BUCC_folder", default=None, type=str)
  parser.add_argument("--FLORES_folder", default=None, type=str)
  parser.add_argument("--eval_folder", type=str)
  args = parser.parse_args()

  if not P.exists(args.eval_folder):
    os.mkdir(args.eval_folder)

  eval_folder = P.join(args.eval_folder, args.model)
  os.mkdir(eval_folder)
  
  if args.model == "labse":
    model = Labse()
    pred_f = lambda x: model.predict(x, args.batch_size, False)
  elif args.model == "word_emb":
    model = WordEmb()
    pred_f = lambda x: model.predict(x, args.batch_size, False)
  else:
    print(f"Model {args.model} does not exist!", file=sys.stderr)

  if args.BUCC_folder is not None:
    BUCC_out_f = P.join(eval_folder, "BUCC")
    BUCC_eval(args.BUCC_folder, BUCC_out_f, pred_f, False, args.use_gpu)

  if args.FLORES_folder is not None:
    FLORES_out_f = P.join(eval_folder, "FLORES")
    FLORES_eval(args.FLORES_folder, FLORES_out_f, pred_f, args.use_gpu)
  

if __name__ == "__main__":
  main()