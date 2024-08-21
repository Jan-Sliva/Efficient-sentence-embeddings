import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os.path as P

PATH = ""

class Labse:

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        if self.device == "cpu":
            self.model.cpu()
        else:
            self.model.cuda(self.device)

    def predict(self, sentences, batch_size, verbose=False):
        return self.model.encode(sentences, batch_size=batch_size, show_progress_bar=verbose)


labse = Labse()

with open(P.join("download_data", "langs.txt"), "r") as f:
    langs = f.readlines()

langs = list(map(lambda x: x.rstrip(), langs))


for lang in langs:
    input_file = "{}/texts/{}.txt".format(PATH, lang)

    with open(input_file, "r") as f:
        inputs = f.readlines()

    inputs = list(map(lambda x: x.rstrip(), inputs))
    if inputs[-1] == "": inputs.pop()


    rets = labse.predict(inputs, 32)
    rets = rets.astype(np.float16)

    np.save(PATH + "/labse_embs/" + lang, rets)


