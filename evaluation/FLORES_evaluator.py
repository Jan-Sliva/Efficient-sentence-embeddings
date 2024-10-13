"""
This file contains the FLORESEvaluator class for evaluating the model on the FLORES+ dataset.
"""
import numpy as np
import os
import time
import faiss
import os.path as P
from evaluation.base_evaluator import BaseEvaluator
from evaluation.utils_retrieve import knn

class FLORESEvaluator(BaseEvaluator):
    @property
    def name(self):
        return "FLORES"

    def __init__(self, **params):
        with open(P.join("evaluation", "labse_langs_FLORES.txt"), "r") as f:
            self.languages_from = [lang.strip() for lang in f.readlines()]
        self.languages_to = params.get("languages_to", ["eng_Latn"])

        self.input_folder = P.join(params["input_folder"], "devtest")
        self.use_gpu = params.get("use_gpu", True)

        self.batch_size = params["batch_size"]
        self.verbose = params.get("verbose", False)

    def evaluate(self, retrieval_model): 
        embs, times = self._extract_embeddings(retrieval_model)
        accuracies = self._calculate_accuracies(embs)

        results = {}
        for lang, time in zip(self.languages_from, times):
            results[f"time_{lang}"] = time

        for lang_from, acc_dict in accuracies.items():
            for lang_to, acc in acc_dict.items():
                results[f"accuracy_{lang_from}_{lang_to}"] = acc

        return results

    def _extract_embeddings(self, retrieval_model):
        embs = []
        times = []
        for lang in self.languages_from:
            file = P.join(self.input_folder, f"devtest.{lang}")
            with open(file, encoding="utf-8") as f:
                sent_list = [l.rstrip("\n") for l in f]

            start = time.time()
            l_emb = retrieval_model.predict(sent_list, self.batch_size, self.verbose)
            end = time.time()

            faiss.normalize_L2(l_emb)
            embs.append(l_emb)
            times.append(end - start)
        return embs, times

    def _calculate_accuracies(self, embs):
        accuracies = {lang_from: {} for lang_from in self.languages_from}

        for i, lang_from in enumerate(self.languages_from):
            for j, lang_to in enumerate(self.languages_to):
                if lang_from == lang_to:
                    continue
                print(f"{lang_from} -> {lang_to}", flush=True)
                acc = self._accuracy_eval(embs[i], embs[self.languages_from.index(lang_to)])
                accuracies[lang_from][lang_to] = acc

        return accuracies

    def _accuracy_eval(self, x: np.array, y: np.array):
        assert x.shape[0] == y.shape[0], "source and target files have different number of elements"
        N = x.shape[0]
        _, ind = knn(x, y, 1, self.use_gpu)
        no_correct = sum(i == ind[i][0] for i in range(N))
        return no_correct / N