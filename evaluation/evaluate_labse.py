from labse import Labse
from evaluate_BUCC import BUCC_eval

import argparse

def main():
  parser = argparse.ArgumentParser(description='LaBse evaluation')
  parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
  parser.add_argument("--BUCC_folder", type=str, default="")
  parser.add_argument("--word_emb_eval_folder", type=str)
  parser.add_argument("--LaBSE_eval_folder", type=str)
  parser.add_argument("--use_gpu", type=bool, default=True)
  args = parser.parse_args()

  model = Labse()

  print(BUCC_eval(args.BUCC_folder, args.word_emb_eval_folder, lambda x: model.predict_word_emb(x, args.batch_size), use_gpu=args.use_gpu))
  print(BUCC_eval(args.BUCC_folder, args.LaBSE_eval_folder, lambda x: model.predict_transformer(x, args.batch_size), use_gpu=args.use_gpu))


if __name__ == "__main__":
  main()