import numpy as np
import os.path as P
from torch.utils.data import Dataset
from bisect import bisect

class DestillationDataset(Dataset):
    def __init__(self, token_folder, emb_folder):
        with open("langs.txt", "r") as f:
            self.langs = f.readlines()
        self.langs = list(map(lambda x: x.rstrip(), self.langs))

        self.token_folder = token_folder
        self.emb_folder = emb_folder

        self.prefix_sum_lengths = [0]

        self.embs = []
        self.tokens = []
        self.tokens_starts = []

        for l in self.langs:
            self.embs.append(DestillationDataset.load_npy_as_mmap(P.join(self.emb_folder, l + ".npy")))
            self.tokens.append(DestillationDataset.load_npy_as_mmap(P.join(self.token_folder, l + ".npy")))
            self.tokens_starts.append(DestillationDataset.load_npy_as_mmap(P.join(self.token_folder, l + "-starts.npy")))

            self.prefix_sum_lengths.append(self.prefix_sum_lengths[-1] + np.shape(self.embs[-1])[0])

    def load_npy_as_mmap(file_name):
        return np.load(file_name, mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return self.prefix_sum_lengths[-1]

    def __getitem__(self, idx):
        lang_idx = bisect(self.prefix_sum_lengths, idx)-1
        sent_idx = idx - self.prefix_sum_lengths[lang_idx]

        tokens = np.copy(self.tokens[lang_idx][self.tokens_starts[lang_idx][sent_idx]:self.tokens_starts[lang_idx][sent_idx+1]])
        emb = np.copy(self.embs[lang_idx][sent_idx])

        return tokens, emb






