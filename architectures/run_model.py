from fairseq.models.lightconv import LightConvEncoder
from torch.nn import Embedding
import numpy as np

embs = Embedding(np.load("BERT-embedings"))

encoder = LightConvEncoder(get_args(), None, embs)




def get_args():
    pass
