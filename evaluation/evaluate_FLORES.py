import numpy as np
import os
import time

import faiss

import os.path as P

from evaluation.utils_retrieve import knn

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

def FLORES_eval(input_folder, output_folder, extract_emb_f, name_of_model, use_gpu=True):

    if not P.exists(output_folder):
        os.mkdir(output_folder)

    with open(P.join("evaluation", "labse_langs_FLORES.txt"), "r") as f:
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

        
    f_times = P.join(output_folder, "{}-FLORES-time.csv".format(name_of_model))
    with open(f_times, "w") as f:
        f.write("language,time\n")
        for i in range(len(languages)):
            f.write("{},{}\n".format(languages[i], times[i]))
        f.write("all,{}\n".format(sum(times)))

    languages_from = languages
    languages_to = ["eng_Latn"]

    averages = [0] * len(languages_to)

    accuracies = []
    for i in range(len(languages_from)):
        accuracies.append([])
        for j in range(len(languages_to)):
            print(languages_from[i] + " -> " + languages_to[j], flush=True)
            accuracies[-1].append(accuracy_eval(embs[languages.index(languages_from[i])], embs[languages.index(languages_to[j])], use_gpu))
            averages[j] += accuracies[i][j]

    for j in range(len(languages_to)):
        averages[j] /= len(languages_from)

    f_accuracies = P.join(output_folder, "{}-FLORES-accuracy.csv".format(name_of_model))
    with open(f_accuracies, "w") as f:
        for lang in languages_to:
            f.write("," + lang)
        f.write("\n")
        for i in range(len(languages_from)):
            f.write(languages_from[i])
            for j in range(len(languages_to)):
                f.write(",")
                f.write("{:.4f}".format(accuracies[i][j]))
            f.write("\n")
        
        f.write("avg")
        for j in range(len(languages_to)):
            f.write(",")
            f.write("{:.4f}".format(averages[j]))
        f.write("\n")
        


    return languages, times, accuracies


