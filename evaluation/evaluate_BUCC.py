import numpy as np
import os

import time

from utils_retrieve import bucc_eval, extract_ids_and_sentences, mine_bitext, extract_file_as_list_bucc
import os.path as P

# BUCC - F1
# from xtreme
BUCC_pairs = ["ru-en", "zh-en", "fr-en", "de-en"]


def pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file=None, use_gpu=False):
    x_file_id = x_file + ".id"
    x_file_sent = x_file + ".sent"

    extract_ids_and_sentences(x_file, x_file_id, x_file_sent)

    y_file_id = y_file + ".id"
    y_file_sent = y_file + ".sent"

    extract_ids_and_sentences(y_file, y_file_id, y_file_sent)

    mine_bitext(x, y, x_file_id, y_file_id, output_file, use_gpu=use_gpu)

    vystup = bucc_eval(output_file, gold_file, x_file_sent, y_file_sent, x_file_id, y_file_id, predict_file)

    os.remove(x_file_id)
    os.remove(x_file_sent)
    os.remove(y_file_id)
    os.remove(y_file_sent)
    
    return vystup


def BUCC_eval(input_folder, output_folder, extract_emb_f, save_embs=False, use_gpu=False):
    vystupy = {}

    if not P.exists(output_folder):
        os.mkdir(output_folder)

    result_file = P.join(output_folder, "results.csv")

    with open(result_file, "w") as f:
        f.write("pair,time,precision,recall,F1,best-threshold\n")

    for pair in BUCC_pairs:
        x_lang = pair.split("-")[0]
        y_lang = pair.split("-")[1]

        x_file = P.join(input_folder, pair, "{}.training.{}".format(pair, x_lang))
        y_file = P.join(input_folder, pair, "{}.training.{}".format(pair, y_lang))
        gold_file =  P.join(input_folder, pair, "{}.training.gold".format(pair))
        output_file = P.join(output_folder, pair, "{}.training.output".format(pair))
        predict_file = P.join(output_folder, pair, "{}.training.predict".format(pair))

        x_list = extract_file_as_list_bucc(x_file)
        y_list = extract_file_as_list_bucc(y_file)

        pair_output_dir = P.join(output_folder, pair)
        if not P.exists(pair_output_dir):
            os.mkdir(pair_output_dir)
        
        start = time.time()
        x = extract_emb_f(x_list)
        y = extract_emb_f(y_list)
        end = time.time()

        if save_embs:
            emb_x_file = P.join(pair_output_dir, "{}.emb".format(x_lang))
            emb_y_file = P.join(pair_output_dir, "{}.emb".format(y_lang))
            np.save(emb_x_file, x)
            np.save(emb_y_file, y)

        vystup = pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file, use_gpu=use_gpu)

        with open(result_file, "a") as f:
            f.write("{},{},{},{},{},{}\n".format(pair, end-start, vystup["precision"], vystup["recall"], vystup["F1"], vystup["best-threshold"]))

        vystupy[pair] = {}
        for key in vystup.keys(): vystupy[pair][key] = vystup[key]
        vystupy[pair]["time"] = end-start

    return vystupy
