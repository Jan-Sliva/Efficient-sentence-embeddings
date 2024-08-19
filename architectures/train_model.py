import torch
from load_data import DestillationDataset
import os.path as P
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, loader, optimizer, loss_fn, epoch_index, tb_writer, report_each=100):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)["encoder_out"]

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % report_each == (report_each-1):
            last_loss = running_loss / report_each # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(model, data_folder, save_folder, tb_folder, lr, batch_size, epochs=1):

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def collate_fn_padd(batch):
        input = [ torch.Tensor(t[0]).to("cuda", dtype=torch.int32) for t in batch ]
        input = torch.nn.utils.rnn.pad_sequence(input, padding_value=0, batch_first=True)

        output = [ torch.Tensor(t[1]).to("cuda", dtype=torch.float32) for t in batch ]
        output = torch.stack(output, dim=0)

        print(input.shape)
        print(output.shape)
        return input, output

    loader = torch.utils.data.DataLoader(DestillationDataset(P.join(data_folder, "tokens"), P.join(data_folder, "labse_embs")),
                         batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padd)
    
    writer = SummaryWriter(tb_folder)

    model.train()

    for e in range(epochs):
        print("Epoch {}".format(e))
        train_one_epoch(model, loader, optimizer, loss_fn, e, writer)
        torch.save(model.state_dict(), P.join(save_folder, "model-{}.pt".format(e+1)))
