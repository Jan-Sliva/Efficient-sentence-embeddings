from labse import Labse
from word_emb import WordEmb
from evaluate_BUCC import BUCC_eval
import json

import os.path as P

import argparse

def main():
  parser = argparse.ArgumentParser(description='LaBse evaluation')
  parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
  parser.add_argument("--BUCC_folder", type=str, default="")
  parser.add_argument("--word_emb_eval_folder", default=None, type=str)
  parser.add_argument("--labse_eval_folder",  default=None, type=str)
  parser.add_argument("--use_gpu", type=bool, default=True)
  args = parser.parse_args()

  def eval_and_save(folder, model):
    ret = BUCC_eval(args.BUCC_folder, folder, lambda x: model.predict(x, args.batch_size, False), use_gpu=args.use_gpu)
    with open(P.join(folder, "results.json"), 'w') as fp:
      json.dump(ret, fp)

  if (args.word_emb_eval_folder != None):
    word_emb = WordEmb()
    eval_and_save(args.word_emb_eval_folder, word_emb)
  
  if (args.labse_eval_folder != None):
    labse = Labse()
    eval_and_save(args.labse_eval_folder, labse)


if __name__ == "__main__":
  main()