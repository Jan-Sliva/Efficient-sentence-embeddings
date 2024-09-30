"""
This script saves the tokens of the News Crawl dataset for each language. The tokens are saved in a one long vector containing all the sentence tokens in a concatenated manner.
I also save a start map, which is a list of the starting indices of the first token for each sentence.

usage:
python save_tokens.py --path <path_to_data_folder>
"""
from transformers import BertTokenizerFast
import numpy as np
import os.path as P
import argparse

def main():
    parser = argparse.ArgumentParser(description='Save tokens of the News Crawl dataset for each language.')
    parser.add_argument('--path', type=str, required=True, help='Path to data folder containing downloaded text data in <path>/texts/. The output tokens will be saved in <path>/tokens/')
    args = parser.parse_args()

    tok = BertTokenizerFast.from_pretrained("setu4993/LaBSE")

    with open(P.join("download_data", "langs.txt"), "r") as f:
        langs = f.readlines()

    langs = list(map(lambda x: x.rstrip(), langs))

    for l in langs:
        print("Processing language {}".format(l))
        input_file = P.join(args.path, f"{l}.txt")
        output_file = P.join(args.path, f"{l}.npy")
        starts_file = P.join(args.path, f"{l}-starts.npy")

        with open(input_file, "r") as f:
            inputs = f.readlines()

        inputs = list(map(lambda x: x.rstrip(), inputs))
        if inputs[-1] == "": inputs.pop()

        output_dict = tok(inputs, return_tensors="np", padding=False, truncation=True)
        tokens = output_dict["input_ids"]

        start_map = [0]
        for i in range(tokens.shape[0]):
            start_map.append(start_map[-1] + tokens[i].shape[0])

        tokens = np.concatenate(tokens).astype(np.int32)

        np.save(output_file, tokens)
        np.save(starts_file, np.array(start_map, dtype=np.int32))
