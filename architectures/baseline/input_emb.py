"""
This module provides a model which averages input embeddings of pretrained Labse model.
"""
import torch
from transformers import BertModel, BertTokenizerFast
from math import ceil
from architectures.base_retrieval_model import BaseRetrievalModel
import numpy as np

class InputEmbAverageModel(BaseRetrievalModel):

    def __init__(self, **params):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = BertModel.from_pretrained("setu4993/LaBSE").to(self.device)
        self.model = self.model.eval()

        inputs = self.tokenizer([""], return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            self.empty_emb = self.model.embeddings.word_embeddings(inputs["input_ids"]).sum(axis=1)
        self.empty_tokens = inputs["attention_mask"].sum(axis=1)

    @torch.inference_mode()
    def predict(self, sentences, batch_size, verbose = False):

        all_embs = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for s in range(0, len(sentences), batch_size):
            e = min(s+batch_size, len(sentences))

            inputs = self.tokenizer(sentences_sorted[s:e], return_tensors="pt", padding=True)
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                fast_emb_sent = self.model.embeddings.word_embeddings(inputs["input_ids"])

            ret = (fast_emb_sent * inputs["attention_mask"][:, :, None]).sum(axis=1) - self.empty_emb
            ret = torch.div(ret, (inputs["attention_mask"].sum(axis=1) - self.empty_tokens)[:, None])
            ret = ret.cpu().detach().numpy()

            all_embs.extend(ret)

        all_embs = [all_embs[idx] for idx in np.argsort(length_sorted_idx)]

        return np.array(all_embs)
    
    def load_weights(self):
        pass

    def train(self):
        pass

    def inference(self, sentences, batch_size, verbose=False):
        return self.predict(sentences, batch_size, verbose)

    
if __name__ == "__main__":
    model = InputEmbAverageModel()
    english_sentences = [
        "dog",
        "Puppies are nice.",
        "I enjoy taking long walks along the beach with my dog.",
    ]
    print(model.predict(english_sentences, 2, True))