## Efficient sentence embedings
Sentence vector representations, also called sentence embeddings, are nowadays used to transfer simpler classification tasks between languages (we have training data in only one language but need the model to work in multiple languages) and to find parallel training sentences for machine translation. Sentence embeddings are typically the output of a Transformer neural network, so computing such embeddings takes a relatively long time. Moreover, efficient computation requires GPU. The goal of this work will be to develop methods to obtain sentence embeddings more efficiently by using simpler neural networks that learn by knowledge distillation. These simpler networks will include models with 1D convolutions.

Embedings will be evaluated on BUCC2018 and FLORES+ datasets. The pytorch library will be used for implementation.

## Conda enviroment

## Usage

### Download the data

Download the training data:
```bash
python download_data/download.py --path <path_to_data_folder>
python download_data/save_labse_embs.py --path <path_to_data_folder>
python download_data/save_tokens.py --path <path_to_data_folder>
```

Save the labse embedding matrix:
```bash
python download_data/save_labse_emb_matrix.py --path <path_to_file_with_labse_emb_matrix>
```

Download the test data from https://github.com/openlanguagedata/flores and https://comparable.limsi.fr/bucc2018/bucc2018-task.html

The folder structure of BUCC2018 data should be:
```
<path_to_BUCC_data>/de-en/
<path_to_BUCC_data>/de-en/de-en.training.en
<path_to_BUCC_data>/de-en/de-en.training.de
<path_to_BUCC_data>/de-en/de-en.training.gold
<path_to_BUCC_data>/fr-en/
<path_to_BUCC_data>/fr-en/fr-en.training.en
<path_to_BUCC_data>/fr-en/fr-en.training.fr
<path_to_BUCC_data>/fr-en/fr-en.test.gold
<path_to_BUCC_data>/zh-en/
...
<path_to_BUCC_data>/ru-en/
...
```
FLORES+ data should be saved in the following folder structure:
```
<path_to_FLORES_data>/devtest/
<path_to_FLORES_data>/devtest/devtest.eng_Latn
<path_to_FLORES_data>/devtest/devtest.ces_Latn
<path_to_FLORES_data>/devtest/devtest.deu_Latn
<path_to_FLORES_data>/devtest/devtest.fin_Latn
<path_to_FLORES_data>/devtest/devtest.fra_Latn
<path_to_FLORES_data>/devtest/devtest.hrv_Latn
<path_to_FLORES_data>/devtest/devtest.ita_Latn
...
```

### Train the model

```bash
python architectures/init_and_train_model.py --data_path <path_to_data_folder> --save_path <path_to_save_folder>  --emb_path <path_to_file_with_labse_emb_matrix>
```
Training data should be saved in `<path_to_data_folder>`.

See tensorboard logs:
```bash
tensorboard --logdir <path_to_save_folder>/tb
```
The weights of the model after each epoch will be saved in `<path_to_save_folder>/save/model-<epoch>.pt`.

### Evaluate the model
Evaluate the model on BUCC2018 and FLORES+ datasets. The results will be saved in `<path_to_eval_folder>`.
```bash
python evaluation/evaluate.py --model light_convolution --model_path <path_to_model_weights> --BUCC_folder <path_to_BUCC_data> --FLORES_folder <path_to_FLORES_data> --eval_folder <path_to_eval_folder>
```
Now evaluate labse model as baseline:
```bash
python evaluation/evaluate.py --model labse --BUCC_folder <path_to_BUCC_data> --FLORES_folder <path_to_FLORES_data> --eval_folder <path_to_eval_folder>
```
And now evaluate the other baseline, which averages word embeddings:
```bash
python evaluation/evaluate.py --model word_emb --BUCC_folder <path_to_BUCC_data> --FLORES_folder <path_to_FLORES_data> --eval_folder <path_to_eval_folder>
```

### Inference
