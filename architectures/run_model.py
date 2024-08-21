from fairseq.models.lightconv import LightConvEncoder
import os.path as P
import os
from architectures.utils import *
import architectures.train_model

folder = "/lnet/aic/personal/slivajan/PRO/21_08"
data_folder = "/lnet/aic/personal/slivajan/PRO/training"
emb_path = "/lnet/aic/personal/slivajan/PRO/Labse-embs.pt"
dict = {"layers" : 1, "kernel_sizes" : [31], "conv_type" : "lightweight", "weight_softmax" : True}
lr = 0.0005
batch_size = 128
epochs = 1


def get_model():
    embs = load_embs(emb_path)
    args = get_args(**dict)

    return LightConvEncoder(args, None, embs).to("cuda")


if __name__ == "__main__":
    create_folder(folder)
    create_folder(data_folder)
    save_folder = P.join(folder, "save")
    create_folder(save_folder)
    tb_folder = P.join(folder, "tb")
    create_folder(tb_folder)

    light_encoder = get_model()

    architectures.train_model.train(light_encoder, data_folder, save_folder, tb_folder, lr, batch_size, epochs)
