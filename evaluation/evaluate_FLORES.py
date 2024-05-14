import numpy as np
import os

from utils_retrieve import extract_file_as_list
import os.path as P

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
