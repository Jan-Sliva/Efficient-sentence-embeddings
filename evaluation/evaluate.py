import numpy as np
import os

from utils_retrieve import bucc_eval, extract_ids_and_sentences, mine_bitext
import os.path as P

class ModelToEvaluate:

    # input: file with multiple lines of text
    # output: list of embs for each line
    def extract_embs(file: str) -> np.array:
        pass

def similarity(a: np.array, b: np.array) -> float: #return sim of two vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# FLORES - accuracy
# gold labels are (0, 0), (1, 1), ... (i, i), ...
def accuracy_eval(x: np.array, y: np.array):

    assert x.shape[0] == y.shape[0], "source and target files have different number of elements"

    N = x.shape[0]

    sim_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            sim_matrix[i, j] = similarity(x[i], y[j])

    no_correct = 0
    for i in range(N):
        if i == np.argmax(sim_matrix[i, :]):
            no_correct += 1
        
    return no_correct/N

# BUCC - F1
# from xtreme
BUCC_pairs = ["zh-en", "ru-en", "fr-en", "de-en"]



def BUCC_eval(folder, model: ModelToEvaluate):
    vystupy = {}

    for pair in BUCC_pairs:
        x_file = P.join(folder, pair, "{}.training.{}".format(pair, pair.split("-")[0]))
        y_file = P.join(folder, pair, "{}.training.{}".format(pair, pair.split("-")[1]))
        gold_file =  P.join(folder, pair, "{}.training.gold".format(pair))
        output_file = P.join(folder, pair, "{}.training.output".format(pair))
        predict_file = P.join(folder, pair, "{}.training.predict".format(pair))

        x = model.extract_embs(x_file)
        y = model.extract_embs(y_file)

        vystup = pair_retrieval_eval(x, y, x_file, y_file, gold_file, output_file, predict_file)
        vystupy[pair] = vystup

    return vystupy



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

N = 10
x = np.random.normal(0, 1, (N, 32)).astype(np.float32)
y = np.random.normal(0, 1, (N, 32)).astype(np.float32)

print(accuracy_eval(x, y))

