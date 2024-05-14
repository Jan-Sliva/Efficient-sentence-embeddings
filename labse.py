import torch
from transformers import BertModel, BertTokenizerFast


class Labse:

    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = BertModel.from_pretrained("setu4993/LaBSE")
        self.model = self.model.eval()

    def predict_transformer(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.pooler_output


    def predict_word_emb(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        fast_emb_sent = self.model.embeddings.word_embeddings(inputs["input_ids"])

        fast_emb = torch.zeros_like(fast_emb_sent[:, 0])

        for i in range(fast_emb_sent.shape[0]):
            ret = torch.zeros_like(fast_emb_sent[0, 0])
            j = 1
            while inputs["input_ids"][i, j] != 102: # 102 = SEP token
                ret += fast_emb_sent[i, j]
                j += 1
            if (j > 1): ret /= (j-1)
            fast_emb[i] = ret

        return fast_emb
    

model = Labse()
english_sentences = [
    "dog",
    "Puppies are nice.",
    "I enjoy taking long walks along the beach with my dog.",
]
print(model.predict_transformer(english_sentences))
print(model.predict_word_emb(english_sentences))