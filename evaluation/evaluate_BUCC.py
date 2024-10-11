import numpy as np
import os
import time
import os.path as P
from evaluation.base_evaluator import BaseEvaluator
from evaluation.utils_retrieve import bucc_eval, extract_ids_and_sentences, mine_bitext, extract_file_as_list_bucc

class BUCCEvaluator(BaseEvaluator):
    @property
    def name(self):
        return "BUCC"

    def __init__(self):
        self.BUCC_pairs = ["ru-en", "zh-en", "fr-en", "de-en"]

    def evaluate(self, input_folder, output_folder, extract_emb_f, use_gpu=True, save_embs=False, save_pairs=False):
        if not P.exists(output_folder):
            os.makedirs(output_folder)

        result_file = P.join(output_folder, f"{self.name}-results.csv")
        pred_folder = P.join(output_folder, f"{self.name}-predictions")
        
        self._initialize_result_file(result_file)
        if not P.exists(pred_folder):
            os.makedirs(pred_folder)

        vystupy = {}
        for pair in self.BUCC_pairs:
            pair_results = self._evaluate_pair(input_folder, pred_folder, pair, extract_emb_f, save_embs, save_pairs, use_gpu)
            self._write_pair_results(result_file, pair, pair_results)
            vystupy[pair] = pair_results

        if not os.listdir(pred_folder):
            os.rmdir(pred_folder)

        return vystupy

    def _initialize_result_file(self, result_file):
        with open(result_file, "w") as f:
            f.write("pair,time,precision,recall,F1,best-threshold\n")

    def _evaluate_pair(self, input_folder, pred_folder, pair, extract_emb_f, save_embs, save_pairs, use_gpu):
        x_lang, y_lang = pair.split("-")
        x_file = P.join(input_folder, pair, f"{pair}.training.{x_lang}")
        y_file = P.join(input_folder, pair, f"{pair}.training.{y_lang}")
        gold_file = P.join(input_folder, pair, f"{pair}.training.gold")

        pair_output_dir = P.join(pred_folder, pair)
        if not P.exists(pair_output_dir):
            os.makedirs(pair_output_dir)

        output_file = P.join(pair_output_dir, f"{pair}.training.output")
        predict_file = P.join(pair_output_dir, f"{pair}.training.predict") if save_pairs else None

        x_list = extract_file_as_list_bucc(x_file)
        y_list = extract_file_as_list_bucc(y_file)

        start = time.time()
        x = extract_emb_f(x_list)
        y = extract_emb_f(y_list)
        end = time.time()

        if save_embs:
            self._save_embeddings(pair_output_dir, x_lang, y_lang, x, y)

        vystup = self._pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file, use_gpu)

        if not save_pairs:
            os.remove(output_file)
            if not os.listdir(pair_output_dir):
                os.rmdir(pair_output_dir)

        vystup["time"] = end - start
        return vystup

    def _save_embeddings(self, pair_output_dir, x_lang, y_lang, x, y):
        emb_x_file = P.join(pair_output_dir, f"{x_lang}.emb")
        emb_y_file = P.join(pair_output_dir, f"{y_lang}.emb")
        np.save(emb_x_file, x)
        np.save(emb_y_file, y)

    def _pair_retrieval_eval(self, x, y, x_file, y_file, gold_file, output_file, predict_file, use_gpu):
        x_file_id, x_file_sent = self._extract_ids_and_sentences(x_file)
        y_file_id, y_file_sent = self._extract_ids_and_sentences(y_file)

        mine_bitext(x, y, x_file_id, y_file_id, output_file, use_gpu=use_gpu)

        vystup = bucc_eval(output_file, gold_file, x_file_sent, y_file_sent, x_file_id, y_file_id, predict_file)

        self._cleanup_files(x_file_id, x_file_sent, y_file_id, y_file_sent)
        
        return vystup

    def _extract_ids_and_sentences(self, file):
        file_id = file + ".id"
        file_sent = file + ".sent"
        extract_ids_and_sentences(file, file_id, file_sent)
        return file_id, file_sent

    def _cleanup_files(self, *files):
        for file in files:
            os.remove(file)

    def _write_pair_results(self, result_file, pair, results):
        with open(result_file, "a") as f:
            f.write(f"{pair},{results['time']},{results['precision']},{results['recall']},{results['F1']},{results['best-threshold']}\n")