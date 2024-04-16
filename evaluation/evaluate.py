import numpy as np


class ModelToEvaluate:

    # input: multiple lines of text
    # output: list of embs for each line
    def predict(text: str) -> np.array:
        pass

def similarity(a: np.array, b: np.array) -> float: #return sim of two vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# FLORES/Tatoeba - accuracy
# golf labels are (0, 0), (1, 1), ... (i, i), ...
def accuracy_eval(source_embs: np.array, target_embs: np.array):

    assert source_embs.shape[0] == target_embs.shape[0], "source and target files have different number of elements"

    N = source_embs.shape[0]

    sim_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            sim_matrix[i, j] = similarity(source_embs[i], source_embs[j])

    no_correct = 0
    for i in range(N):
        if i == np.argmax(sim_matrix[i, :]):
            no_correct += 1
        
    return no_correct/N

# BUCC - F1
# from xtreme
def F1_eval():
    