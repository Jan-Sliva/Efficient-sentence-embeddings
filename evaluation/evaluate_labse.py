from labse import Labse
from evaluate_BUCC import BUCC_eval

import argparse

def main():
  args = argparse.ArgumentParser(description='LaBse evaluation')
  args.add_argument("--batch_size", type=int, default=100, help="batch size.")
  args.add_argument("--BUCC_folder", type=str)
  args.add_argument("--word_emb_eval_folder", type=str)
  args.add_argument("--LaBSE_eval_folder", type=str)

  model = Labse()

  print(BUCC_eval(args.BUCC_folder, args.word_emb_eval_folder, lambda x: model.predict_word_emb(x, args.batch_size)))
  print(BUCC_eval(args.BUCC_folder, args.LaBSE_eval_folder, lambda x: model.predict_transformer(x, args.batch_size)))


if __name__ == "__main__":
  main()