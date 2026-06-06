"""
This file contains the TATOEBAEvaluator class for evaluating the model on the TATOEBA dataset.
"""
import numpy as np
import os
import time
import faiss
import os.path as P
from evaluation.base_evaluator import BaseEvaluator
from evaluation.utils_retrieve import knn

class TATOEBAEvaluator(BaseEvaluator):
    @property
    def name(self):
        return "TATOEBA"
    
    def __init__(self, **params):
        self.input_folder = params["input_folder"]
        self.use_gpu = params.get("use_gpu", True)

        self.batch_size = params["batch_size"]
        self.verbose = params.get("verbose", False)

    def _accuracy_eval(self, x: np.array, y: np.array):
        assert x.shape[0] == y.shape[0], "source and target files have different number of elements"
        N = x.shape[0]
        _, ind = knn(x, y, 1, self.use_gpu)
        no_correct = sum(i == ind[i][0] for i in range(N))
        return no_correct / N

    def eval_lang(self, lang, retrieval_model):
        folder = P.join(self.input_folder, lang)

        source_file = P.join(folder, "source.txt")
        target_file = P.join(folder, "target.txt")

        with open(source_file, "r") as f:
            source_lines = [l.strip() for l in f.readlines()]
        with open(target_file, "r") as f:
            target_lines = [l.strip() for l in f.readlines()]

        assert len(source_lines) == len(target_lines)

        start = time.time()
        l_emb = retrieval_model.predict(source_lines, self.batch_size, self.verbose)
        t_emb = retrieval_model.predict(target_lines, self.batch_size, self.verbose)
        end = time.time()

        faiss.normalize_L2(l_emb)
        faiss.normalize_L2(t_emb)

        time_taken = end - start
        accuracy = self._accuracy_eval(l_emb, t_emb)
        
        return accuracy, time_taken

    def evaluate(self, retrieval_model):
        langs_folders = os.listdir(self.input_folder)
        results = {}
        for lang in langs_folders:
            accuracy, time_taken = self.eval_lang(lang, retrieval_model)
            results[f"accuracy_{lang}"] = accuracy
            results[f"time_{lang}"] = time_taken
        return results
    
