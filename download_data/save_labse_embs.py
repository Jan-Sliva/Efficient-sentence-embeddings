"""
This script saves the LaBSE embeddings for each language.

usage:
python download_data/save_labse_embs.py --path <path_to_data_folder>
"""
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os.path as P
import argparse

class Labse:

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        if self.device == "cpu":
            self.model.cpu()
        else:
            self.model.cuda(self.device)

    def predict(self, sentences, batch_size, verbose=False):
        return self.model.encode(sentences, batch_size=batch_size, show_progress_bar=verbose)

def main():
    parser = argparse.ArgumentParser(description='Save LaBSE embeddings for each language.')
    parser.add_argument('--path', type=str, required=True, help='Path to data folder containing downloaded text data in <path>/texts/. \
                                                                The output embeddings will be saved in <path>/labse_embs/')
    args = parser.parse_args()

    labse = Labse()

    with open(P.join("download_data", "langs.txt"), "r") as f:
        langs = f.readlines()

    langs = list(map(lambda x: x.rstrip(), langs))


    for lang in langs:
        input_file = "{}/texts/{}.txt".format(args.path, lang)

        with open(input_file, "r") as f:
            inputs = f.readlines()

        inputs = list(map(lambda x: x.rstrip(), inputs))
        if inputs[-1] == "": inputs.pop()


        rets = labse.predict(inputs, 32)
        rets = rets.astype(np.float16)

        np.save(args.path + "/labse_embs/" + lang, rets)


if __name__ == "__main__":
    main()
