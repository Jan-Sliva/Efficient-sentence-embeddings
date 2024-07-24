import numpy as np
import os
import time

import word_emb
import labse

import faiss

import os.path as P

from utils_retrieve import knn

# FLORES - accuracy
# gold labels are (0, 0), (1, 1), ... (i, i), ...
def accuracy_eval(x: np.array, y: np.array, use_gpu):

    assert x.shape[0] == y.shape[0], "source and target files have different number of elements"

    N = x.shape[0]

    _, ind = knn(x, y, 1, use_gpu)

    no_correct = 0
    for i in range(N):
        if i == ind[i][0]:
            no_correct += 1
        
    return no_correct/N

def FLORES_eval(input_folder, output_folder, extract_emb_f, use_gpu=True):

    if not P.exists(output_folder):
        os.mkdir(output_folder)

    with open("labse_langs_FLORES.txt", "r") as f:
        languages = f.readlines()
    languages = map(lambda x: x.rstrip(), languages)

    languages = list(languages)

    input_folder = P.join(input_folder, "devtest")

    embs = []
    times = []

    
    for lang in languages:
        file = P.join(input_folder, "devtest." + lang)
        with open(file, encoding="utf-8") as f:
            sent_list = [l.rstrip("\n") for l in f]

        start = time.time()
        l_emb = extract_emb_f(sent_list)
        end = time.time()
    
        faiss.normalize_L2(l_emb)

        embs.append(l_emb)
        times.append(end-start)

        
    f_times = P.join(output_folder, "time.csv")
    with open(f_times, "w") as f:
        f.write("language,time\n")
        for i in range(len(languages)):
            f.write("{},{}\n".format(languages[i], times[i]))
        f.write("all,{}\n".format(sum(times)))


    accuracies = []
    for i in range(len(languages)):
        accuracies.append([])
        for j in range(len(languages)):
            print(languages[i] + " - " + languages[j], flush=True)
            accuracies[-1].append(accuracy_eval(embs[i], embs[j], use_gpu))

    f_accuracies = P.join(output_folder, "accuracy.csv")
    with open(f_accuracies, "w") as f:
        for lang in languages:
            f.write("," + lang)
        f.write("\n")
        for i in range(len(languages)):
            f.write(languages[i])
            for j in range(len(languages)):
                f.write(",")
                f.write("{:.4f}".format(accuracies[i][j]))
            f.write("\n")


    return languages, times, accuracies


