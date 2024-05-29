from labse import Labse
from word_emb import WordEmb
from evaluate_BUCC import BUCC_eval

import argparse

def main():
  parser = argparse.ArgumentParser(description='LaBse evaluation')
  parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
  parser.add_argument("--BUCC_folder", type=str, default="")
  parser.add_argument("--word_emb_eval_folder", default=None, type=str)
  parser.add_argument("--labse_eval_folder",  default=None, type=str)
  parser.add_argument("--use_gpu", type=bool, default=True)
  args = parser.parse_args()

  labse = Labse()
  word_emb = WordEmb()

  if (args.word_emb_eval_folder != None):
    print(BUCC_eval(args.BUCC_folder, args.word_emb_eval_folder, lambda x: word_emb.predict(x, args.batch_size, False), use_gpu=args.use_gpu))
  
  if (args.labse_eval_folder != None):
    print(BUCC_eval(args.BUCC_folder, args.labse_eval_folder, lambda x: labse.predict(x, args.batch_size, True), use_gpu=args.use_gpu))


if __name__ == "__main__":
  main()