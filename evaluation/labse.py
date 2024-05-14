import torch
from transformers import BertModel, BertTokenizerFast

class Labse:

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = BertModel.from_pretrained("setu4993/LaBSE").to(self.device)
        self.model = self.model.eval()

    def predict_transformer(self, sentences, batch_size):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        emb_dim = self.model.pooler.dense.weight.size()[0]
        ret = torch.zeros((len(sentences), emb_dim))

        for s in range(0, len(sentences), batch_size):
            e = min(s+batch_size, len(sentences))
            inputs_batch = {}
            for key in inputs.keys():
                inputs_batch[key] = inputs[key][s:e].to(self.device)
            with torch.no_grad():
                ret[s:e] = self.model(**inputs_batch).pooler_output

        return ret.detach().numpy()


    def predict_word_emb(self, sentences, batch_size):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        emb_dim = self.model.embeddings.word_embeddings.embedding_dim

        fast_emb = torch.zeros((len(sentences), emb_dim))
        inputs["input_ids"] = inputs["input_ids"].to(self.device)

        for s in range(0, len(sentences), batch_size):
            e = min(s+batch_size, len(sentences))
            fast_emb_sent = self.model.embeddings.word_embeddings(inputs["input_ids"][s:e])

            for i in range(fast_emb_sent.shape[0]):
                fast_emb[s+i] = torch.zeros((emb_dim,))
                j = 1
                while inputs["input_ids"][s+i, j] != 102: # 102 = SEP token
                    fast_emb[s+i] += fast_emb_sent[i, j]
                    j += 1
                if (j > 1): fast_emb[s+i] /= (j-1)

            del fast_emb_sent

        return fast_emb.detach().numpy()
    
if __name__ == "__main__":
    model = Labse()
    english_sentences = [
        "dog",
        "Puppies are nice.",
        "I enjoy taking long walks along the beach with my dog.",
    ]
    print(model.predict_transformer(english_sentences, 2))
    print(model.predict_word_emb(english_sentences, 2))