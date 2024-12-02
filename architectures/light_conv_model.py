"""
This file contains the LightConvModel class for distillation training, loading weights, and inference.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from fairseq.models.lightconv import LightConvEncoder, base_architecture
from architectures.load_data import DestillationDataset
from architectures.utils import load_embs
from architectures.base_retrieval_model import BaseRetrievalModel
import os.path as P
import os
from argparse import Namespace
from transformers import BertTokenizerFast
from math import ceil
import json
import numpy as np

class LightConvModel(BaseRetrievalModel):
    def __init__(self, **params):
        self.params = params

        self.embs = load_embs(params["emb_path"])
        args = self._get_args(params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LightConvEncoder(args, None, self.embs).to(self.device)
        self.loss_fn = None
        self.tokenizer = None

    def _get_args(self, params):
        """
        Sets the arguments for the LightConv model.
        """
        args = Namespace()
        args.encoder_layers = params.get("layers", 1)
        args.encoder_kernel_size_list = params.get("kernel_sizes", [31])
        args.encoder_conv_type = params.get("conv_type", "lightweight")
        args.weight_softmax = params.get("weight_softmax", True)
        args.encoder_embed_dim = 768
        args.max_source_positions = 1024
        base_architecture(args)
        return args
    
    def _get_outputs(self, inputs, attention_mask):
        outputs = self.model(inputs)["encoder_out"][0]
        outputs = (attention_mask * outputs).sum(dim=1) / attention_mask.sum(dim=1)
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return outputs

    def train(self):
        data_folder = self.params["data_folder"]
        save_folder = P.join(self.params["save_folder"], self.params["name"])
        tb_folder = self.params["tb_folder"]

        for folder in [save_folder, tb_folder, data_folder]:
            if not P.exists(folder):
                os.makedirs(folder)

        if self.loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        train_loader, val_loader = self._get_data_loaders(data_folder, self.params["batch_size"], self.params["val_sentences"])
        writer = SummaryWriter(tb_folder)

        self.model.train()

        best_loss = float("inf")
        best_model_path = None

        weights_folder = P.join(save_folder, "weights")
        if not P.exists(weights_folder):
            os.makedirs(weights_folder)

        for e in range(self.params["epochs"]):
            print(f"Epoch {e}", flush=True)
            last_loss = self._train_one_epoch(train_loader, val_loader, optimizer, e, writer)
            if last_loss < best_loss:
                best_loss = last_loss
                best_model_path = P.join(weights_folder, f"{e+1:03d}.pt")
            torch.save(self.model.state_dict(), best_model_path)

        self.params["best_model_path"] = best_model_path

        json.dump(self.params, open(P.join(save_folder, "params.json"), "w"))

    def load_weights(self, weights_path = None):
        if weights_path is None:
            weights_path = self.params["best_model_path"]
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def _train_one_epoch(self, train_loader, val_loader, optimizer, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = self._get_outputs(inputs[0], inputs[1])

            loss = self.loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % self.params["report_each"] == (self.params["report_each"]-1):
                last_loss = running_loss / self.params["report_each"]
                print(f'  batch {i + 1} loss: {last_loss}', flush=True)
                tb_x = epoch_index * len(train_loader) + i + 1
                running_loss = 0.

                # Run validation after every 'report_each' training steps
                val_loss = self._validate(val_loader)
                print(f'  batch {i + 1} validation loss: {val_loss}', flush=True)

                tb_writer.add_scalars(self.params["name"], {"train_loss": last_loss, "validation_loss": val_loss}, tb_x)

            if i >= self.params["percentage"] * len(train_loader):
                break

        return val_loss

    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self._get_outputs(inputs[0], inputs[1])
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.model.train()
        return avg_loss

    def _get_data_loaders(self, data_folder, batch_size, val_sentences):
        def collate_fn_padd(batch):
            input = [torch.Tensor(t[0]).to("cuda", dtype=torch.int32) for t in batch]
            attention_mask = [torch.ones(len(t[0])).to("cuda", dtype=torch.int32) for t in batch]
            input = torch.nn.utils.rnn.pad_sequence(input, padding_value=0, batch_first=True)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)
            input = (input, attention_mask)

            output = [torch.Tensor(t[1]).to("cuda", dtype=torch.float32) for t in batch]
            output = torch.stack(output, dim=0)

            return input, output

        dataset = DestillationDataset(P.join(data_folder, "tokens"), P.join(data_folder, "labse_embs"))
        val_size = min(val_sentences, len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padd)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_padd)

        return train_loader, val_loader

    @torch.inference_mode()
    def predict(self, sentences, batch_size, verbose=False):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")

        all_embs = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for s in range(0, len(sentences), batch_size):
            if verbose:
                print(f"batch no. {1 + s//batch_size}/{ceil(len(sentences)/batch_size)}")
            e = min(s+batch_size, len(sentences))

            inputs = self.tokenizer(sentences_sorted[s:e], return_tensors="pt", padding=True)
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                emb = self._get_outputs(inputs["input_ids"], inputs["attention_mask"])

            emb = emb.cpu().detach().numpy()
            all_embs.extend(emb)

        all_embs = [all_embs[idx] for idx in np.argsort(length_sorted_idx)]
        return np.array(all_embs)
