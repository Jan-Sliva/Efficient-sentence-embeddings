"""
This module provides a wrapper for the LaBSE model.
"""
import torch
from sentence_transformers import SentenceTransformer
from architectures.base_retrieval_model import BaseRetrievalModel


class Labse(BaseRetrievalModel):

    def __init__(self, **params):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        if self.device == "cpu":
            self.model.cpu()
        else:
            self.model.cuda(self.device)

    def predict(self, sentences, batch_size, verbose=False):
        return self.model.encode(sentences, batch_size=batch_size, show_progress_bar=verbose)
    
    def load_weights(self):
        pass

    def train(self):
        pass

    def inference(self, sentences, batch_size, verbose=False):
        return self.predict(sentences, batch_size, verbose)

    
if __name__ == "__main__":
    model = Labse()
    english_sentences = [
        "dog",
        "Puppies are nice.",
        "I enjoy taking long walks along the beach with my dog.",
    ]
    print(model.predict(english_sentences, 2))