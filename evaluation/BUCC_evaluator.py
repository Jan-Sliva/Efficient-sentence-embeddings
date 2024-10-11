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

    def __init__(self, **params):
        self.BUCC_pairs = ["ru-en", "zh-en", "fr-en", "de-en"]
        self.input_folder = params["input_folder"]
        self.output_folder = params["output_folder"]
        self.save_embs = params.get("save_embs", False)
        self.save_pairs = params.get("save_pairs", False)
        self.use_gpu = params.get("use_gpu", True)

        self.batch_size = params["batch_size"]
        self.verbose = params.get("verbose", False)

    def evaluate(self, retrieval_model):

        pred_folder = P.join(self.output_folder, f"{self.name}-predictions")
        if not P.exists(pred_folder):
            os.makedirs(pred_folder)

        vystupy = {}
        for pair in self.BUCC_pairs:
            pair_results = self._evaluate_pair(pred_folder, pair, retrieval_model)
            for key, value in pair_results.items():
                vystupy[f"{pair}_{key}"] = value

        if not os.listdir(pred_folder):
            os.rmdir(pred_folder)

        return vystupy

    def _evaluate_pair(self, pred_folder, pair, retrieval_model):
        x_lang, y_lang = pair.split("-")
        x_file = P.join(self.input_folder, pair, f"{pair}.training.{x_lang}")
        y_file = P.join(self.input_folder, pair, f"{pair}.training.{y_lang}")
        gold_file = P.join(self.input_folder, pair, f"{pair}.training.gold")

        pair_output_dir = P.join(pred_folder, pair)
        if not P.exists(pair_output_dir):
            os.makedirs(pair_output_dir)

        output_file = P.join(pair_output_dir, f"{pair}.training.output")
        predict_file = P.join(pair_output_dir, f"{pair}.training.predict") if self.save_pairs else None


        x_list = extract_file_as_list_bucc(x_file)
        y_list = extract_file_as_list_bucc(y_file)

        start = time.time()
        x = retrieval_model.predict(x_list, self.batch_size, self.verbose)
        y = retrieval_model.predict(y_list, self.batch_size, self.verbose)
        end = time.time()

        if self.save_embs:
            self._save_embeddings(pair_output_dir, x_lang, y_lang, x, y)

        vystup = self._pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file)

        if not self.save_pairs:
            self._cleanup_files(output_file)

            if not os.listdir(pair_output_dir):
                os.rmdir(pair_output_dir)

        vystup["time"] = end - start
        return vystup

    def _save_embeddings(self, pair_output_dir, x_lang, y_lang, x, y):
        emb_x_file = P.join(pair_output_dir, f"{x_lang}.emb")
        emb_y_file = P.join(pair_output_dir, f"{y_lang}.emb")
        np.save(emb_x_file, x)
        np.save(emb_y_file, y)

    def _pair_retrieval_eval(self, x, y, x_file, y_file, gold_file, output_file, predict_file):
        x_file_id, x_file_sent = self._extract_ids_and_sentences(x_file)
        y_file_id, y_file_sent = self._extract_ids_and_sentences(y_file)

        mine_bitext(x, y, x_file_id, y_file_id, output_file, self.use_gpu)

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