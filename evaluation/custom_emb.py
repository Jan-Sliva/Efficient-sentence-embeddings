"""
This module enables evalaution of torch models with pretrained weights
"""
import torch
from transformers import BertTokenizerFast
from math import ceil

class CustomEmb:

    def __init__(self, model, weights_path, emb_dim) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = model
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.emb_dim = emb_dim


    def predict(self, sentences, batch_size, verbose = False):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        embs = torch.zeros((len(sentences), self.emb_dim)).to(self.device)
        inputs["input_ids"] = inputs["input_ids"].to(self.device)

        for s in range(0, len(sentences), batch_size):
            if (verbose): print(f"batch no. {1 + s//batch_size}/{ceil(len(sentences)/batch_size)}")
            e = min(s+batch_size, len(sentences))
            with torch.no_grad():
                embs[s:e] = self.model(inputs["input_ids"][s:e])["encoder_out"][0].mean(dim=0)
                embs[s:e] = torch.nn.functional.normalize(embs[s:e], p=2, dim=1)

        return embs.cpu().detach().numpy()
  