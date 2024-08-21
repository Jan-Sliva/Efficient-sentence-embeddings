from transformers import BertTokenizerFast
import numpy as np
import os.path as P

tok = BertTokenizerFast.from_pretrained("setu4993/LaBSE")

with open(P.join("download_data", "langs.txt"), "r") as f:
    langs = f.readlines()

langs = list(map(lambda x: x.rstrip(), langs))

for l in langs:
    print("Processing language {}".format(l))
    input_file = "/lnet/aic/personal/slivajan/PRO/training/texts/{}.txt".format(l)
    output_file = "/lnet/aic/personal/slivajan/PRO/training/tokens/{}.npy".format(l)
    starts_file = "/lnet/aic/personal/slivajan/PRO/training/tokens/{}-starts.npy".format(l)

    with open(input_file, "r") as f:
        inputs = f.readlines()

    inputs = list(map(lambda x: x.rstrip(), inputs))
    if inputs[-1] == "": inputs.pop()

    output_dict = tok(inputs, return_tensors="np", padding=False, truncation=True)
    tokens = output_dict["input_ids"]

    start_map = [0]
    for i in range(tokens.shape[0]):
        start_map.append(start_map[-1] + tokens[i].shape[0])

    tokens = np.concatenate(tokens).astype(np.int32)

    np.save(output_file, tokens)
    np.save(starts_file, np.array(start_map, dtype=np.int32))
