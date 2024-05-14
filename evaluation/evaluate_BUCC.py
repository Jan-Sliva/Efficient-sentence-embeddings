import numpy as np
import os

from utils_retrieve import bucc_eval, extract_ids_and_sentences, mine_bitext, extract_file_as_list
import os.path as P

# BUCC - F1
# from xtreme
BUCC_pairs = ["zh-en", "ru-en", "fr-en", "de-en"]



def pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file=None):
    x_file_id = x_file + ".id"
    x_file_sent = x_file + ".sent"

    extract_ids_and_sentences(x_file, x_file_id, x_file_sent)

    y_file_id = y_file + ".id"
    y_file_sent = y_file + ".sent"

    extract_ids_and_sentences(y_file, y_file_id, y_file_sent)

    mine_bitext(x, y, x_file_id, y_file_id, output_file)

    vystup = bucc_eval(output_file, gold_file, x_file_sent, y_file_sent, x_file_id, y_file_id, predict_file)

    os.remove(x_file_id)
    os.remove(x_file_sent)
    os.remove(y_file_id)
    os.remove(y_file_sent)
    
    return vystup


def BUCC_eval(input_folder, output_folder, extract_emb_f, save_embs=True):
    vystupy = {}

    for pair in BUCC_pairs:
        x_lang = pair.split("-")[0]
        y_lang = pair.split("-")[1]

        x_file = P.join(input_folder, pair, "{}.training.{}".format(pair, x_lang))
        y_file = P.join(input_folder, pair, "{}.training.{}".format(pair, y_lang))
        gold_file =  P.join(input_folder, pair, "{}.training.gold".format(pair))
        output_file = P.join(output_folder, pair, "{}.training.output".format(pair))
        predict_file = P.join(output_folder, pair, "{}.training.predict".format(pair))

        x_list = extract_file_as_list(x_file, mode="bucc")
        y_list = extract_file_as_list(y_file, mode="bucc")

        emb_x_file = P.join(output_folder, pair, "{}.emb".format(x_lang))
        emb_y_file = P.join(output_folder, pair, "{}.emb".format(y_lang))

        x = extract_emb_f(x_list)
        y = extract_emb_f(y_list)

        if save_embs:
            np.save(emb_x_file, x)
            np.save(emb_y_file, y)

        vystup = pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file)
        vystupy[pair] = vystup

    return vystupy

    

