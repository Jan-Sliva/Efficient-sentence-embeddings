"""
This module provides a model which averages input embeddings of pretrained Labse model.
"""
import torch
from transformers import BertModel, BertTokenizerFast
from math import ceil

class WordEmb:

    def __init__(self) -> None:
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

    def predict(self, sentences, batch_size, verbose = False):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        emb_dim = self.model.embeddings.word_embeddings.embedding_dim

        fast_emb = torch.zeros((len(sentences), emb_dim)).to(self.device)
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        for s in range(0, len(sentences), batch_size):
            if (verbose): print(f"batch no. {1 + s//batch_size}/{ceil(len(sentences)/batch_size)}")
            e = min(s+batch_size, len(sentences))
            with torch.no_grad():
                fast_emb_sent = self.model.embeddings.word_embeddings(inputs["input_ids"][s:e])

            fast_emb[s:e] = (fast_emb_sent * inputs["attention_mask"][s:e][:, :, None]).sum(axis=1) - self.empty_emb
            fast_emb[s:e] = torch.div(fast_emb[s:e], (inputs["attention_mask"][s:e].sum(axis=1) - self.empty_tokens)[:, None])

            del fast_emb_sent

        return fast_emb.cpu().detach().numpy()
    

    
if __name__ == "__main__":
    model = WordEmb()
    english_sentences = [
        "dog",
        "Puppies are nice.",
        "I enjoy taking long walks along the beach with my dog.",
    ]
    print(model.predict(english_sentences, 2, True))