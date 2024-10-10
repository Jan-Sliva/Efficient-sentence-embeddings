"""
This file contains the LightConvModel class for distillation training, loading weights, and inference.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from fairseq.models.lightconv import LightConvEncoder, base_architecture
from architectures.load_data import DestillationDataset
from architectures.utils import load_embs
from architectures.base_distillation_model import BaseDistillationModel
import os.path as P
from argparse import Namespace

class LightConvModel(BaseDistillationModel):
    def __init__(self, emb_path, layers=1, kernel_sizes=[31], conv_type="lightweight", weight_softmax=True):
        self.embs = load_embs(emb_path)
        args = self._get_args(layers, kernel_sizes, conv_type, weight_softmax)
        self.model = LightConvEncoder(args, None, self.embs).to("cuda")
        self.loss_fn = torch.nn.MSELoss()

    def _get_args(self, layers, kernel_sizes, conv_type, weight_softmax):
        """
        Sets the arguments for the LightConv model.

        layers - int
        kernel_sizes - List[int] (default - all 31)
        conv_type - (lightweight|dynamic)
        weight_softmax - bool
        """
        args = Namespace()
        args.encoder_layers = layers
        args.encoder_kernel_size_list = kernel_sizes
        args.encoder_conv_type = conv_type
        args.weight_softmax = weight_softmax
        args.encoder_embed_dim = 768
        args.max_source_positions = 1024
        base_architecture(args)
        return args
        
    def train(self, data_folder, save_folder, tb_folder, lr, batch_size, epochs=1, percentage=1, val_split=0.1):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_loader, val_loader = self._get_data_loaders(data_folder, batch_size, val_split)
        writer = SummaryWriter(tb_folder)

        self.model.train()

        for e in range(epochs):
            print(f"Epoch {e}", flush=True)
            self._train_one_epoch(train_loader, val_loader, optimizer, e, writer, percentage)
            torch.save(self.model.state_dict(), P.join(save_folder, f"model-{e+1}.pt"))

    def load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)["encoder_out"][0]
            output = output.mean(dim=0)
            output = torch.nn.functional.normalize(output, p=2, dim=1)
        return output

    def _train_one_epoch(self, train_loader, val_loader, optimizer, epoch_index, tb_writer, percentage, report_each=100):
        running_loss = 0.
        last_loss = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = self.model(inputs)["encoder_out"][0]
            outputs = outputs.mean(dim=0)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)

            loss = self.loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % report_each == (report_each-1):
                last_loss = running_loss / report_each
                print(f'  batch {i + 1} loss: {last_loss}', flush=True)
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

                # Run validation after every 'report_each' training steps
                val_loss = self._validate(val_loader)
                print(f'  batch {i + 1} validation loss: {val_loss}', flush=True)
                tb_writer.add_scalar('Loss/validation', val_loss, tb_x)

            if i >= percentage * len(train_loader):
                break

        return last_loss

    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)["encoder_out"][0]
                outputs = outputs.mean(dim=0)
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.model.train()
        return avg_loss

    def _get_data_loaders(self, data_folder, batch_size, val_split=0.1):
        def collate_fn_padd(batch):
            input = [torch.Tensor(t[0]).to("cuda", dtype=torch.int32) for t in batch]
            input = torch.nn.utils.rnn.pad_sequence(input, padding_value=0, batch_first=True)

            output = [torch.Tensor(t[1]).to("cuda", dtype=torch.float32) for t in batch]
            output = torch.stack(output, dim=0)

            return input, output

        dataset = DestillationDataset(P.join(data_folder, "tokens"), P.join(data_folder, "labse_embs"))
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padd)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_padd)

        return train_loader, val_loader