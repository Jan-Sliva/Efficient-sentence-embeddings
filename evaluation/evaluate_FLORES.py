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

    def __init__(self):
        with open(P.join("evaluation", "labse_langs_FLORES.txt"), "r") as f:
            self.languages = [lang.strip() for lang in f.readlines()]
        self.languages_to = ["eng_Latn"]

    def evaluate(self, input_folder, output_folder, extract_emb_f, use_gpu=True):
        if not P.exists(output_folder):
            os.makedirs(output_folder)

        input_folder = P.join(input_folder, "devtest")
        embs, times = self._extract_embeddings(input_folder, extract_emb_f)
        
        self._write_times(output_folder, times)
        accuracies, averages = self._calculate_accuracies(embs, use_gpu)
        self._write_accuracies(output_folder, accuracies, averages)

        return self.languages, times, accuracies

    def _extract_embeddings(self, input_folder, extract_emb_f):
        embs = []
        times = []
        for lang in self.languages:
            file = P.join(input_folder, f"devtest.{lang}")
            with open(file, encoding="utf-8") as f:
                sent_list = [l.rstrip("\n") for l in f]

            start = time.time()
            l_emb = extract_emb_f(sent_list)
            end = time.time()

            faiss.normalize_L2(l_emb)
            embs.append(l_emb)
            times.append(end - start)
        return embs, times

    def _write_times(self, output_folder, times):
        f_times = P.join(output_folder, f"{self.name}-time.csv")
        with open(f_times, "w") as f:
            f.write("language,time\n")
            for lang, time in zip(self.languages, times):
                f.write(f"{lang},{time}\n")
            f.write(f"all,{sum(times)}\n")

    def _calculate_accuracies(self, embs, use_gpu):
        accuracies = []
        averages = [0] * len(self.languages_to)
        for i, lang_from in enumerate(self.languages):
            accuracies.append([])
            for j, lang_to in enumerate(self.languages_to):
                print(f"{lang_from} -> {lang_to}", flush=True)
                acc = self._accuracy_eval(embs[i], embs[self.languages.index(lang_to)], use_gpu)
                accuracies[-1].append(acc)
                averages[j] += acc
        averages = [avg / len(self.languages) for avg in averages]
        return accuracies, averages

    def _accuracy_eval(self, x: np.array, y: np.array, use_gpu):
        assert x.shape[0] == y.shape[0], "source and target files have different number of elements"
        N = x.shape[0]
        _, ind = knn(x, y, 1, use_gpu)
        no_correct = sum(i == ind[i][0] for i in range(N))
        return no_correct / N

    def _write_accuracies(self, output_folder, accuracies, averages):
        f_accuracies = P.join(output_folder, f"{self.name}-accuracy.csv")
        with open(f_accuracies, "w") as f:
            f.write("," + ",".join(self.languages_to) + "\n")
            for lang, acc_row in zip(self.languages, accuracies):
                f.write(f"{lang}," + ",".join(f"{acc:.4f}" for acc in acc_row) + "\n")
            f.write("avg," + ",".join(f"{avg:.4f}" for avg in averages) + "\n")