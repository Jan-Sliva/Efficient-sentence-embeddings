"""
This script initializes the LightConvModel, the folder structure and runs the distillation training.
After each epoch the model is saved in "<save_path>/save/model-<epoch>.pt".
The tensorboard logs are saved in "<save_path>/tb/".

usage:
python architectures/init_and_train_model.py --save_path <path_to_save_folder> --data_path <path_to_data_folder> --emb_path <path_to_file_with_labse_emb_matrix> --lr <learning_rate> --batch_size <batch_size> --epochs <number_of_epochs>
"""
import os.path as P
import os
from architectures.light_conv_model import LightConvModel
from architectures.utils import create_folder
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run the distillation training.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the folder')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--emb_path', type=str, required=True, help='Path to file with word embeddings matrix')
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.0005)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
    parser.add_argument('--percentage', type=float, help='Percentage of the data to use in each epoch', default=1)
    return parser.parse_args()

def main():
    args = parse_args()

    folder = args.save_path
    data_folder = args.data_path
    emb_path = args.emb_path
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    percentage = args.percentage

    create_folder(folder)
    save_folder = P.join(folder, "save")
    create_folder(save_folder)
    tb_folder = P.join(folder, "tb")
    create_folder(tb_folder)

    model = LightConvModel(emb_path, layers=1, kernel_sizes=[31], conv_type="lightweight", weight_softmax=True)
    model.train(data_folder, save_folder, tb_folder, lr, batch_size, epochs, percentage)

if __name__ == "__main__":
    main()
