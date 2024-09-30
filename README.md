## Efficient sentence embedings
Sentence vector representations, also called sentence embeddings, are nowadays used to transfer simpler classification tasks between languages (we have training data in only one language but need the model to work in multiple languages) and to find parallel training sentences for machine translation. Sentence embeddings are typically the output of a Transformer neural network, so computing such embeddings takes a relatively long time. Moreover, efficient computation requires GPU. The goal of this work will be to develop methods to obtain sentence embeddings more efficiently by using simpler neural networks that learn by knowledge distillation. These simpler networks will include models with 1D convolutions.

Embedings will be evaluated on BUCC2018 and FLORES+ datasets. The pytorch library will be used for implementation.

## Conda enviroment

## Usage

### Downloading the data

Download the training data using following scripts:
```bash
python download_data/download.py --path <path_to_data_folder>
python download_data/save_labse_embs.py --path <path_to_data_folder>
python download_data/save_tokens.py --path <path_to_data_folder>
```

Downlaod the test data from https://github.com/openlanguagedata/flores and https://comparable.limsi.fr/bucc2018/bucc2018-task.html

### Training the model
<path_to_labse_embs> - file with labse input embeddings

```bash
python architectures/run_model.py --data_path <path_to_data_folder> --save_path <path_to_save_folder>  --emb_path <path_to_labse_embs>
```

### Evaluating the model
<path_to_eval_folder> - path to folder where the results will be saved

```bash
python evaluation/evaluate.py --model <name_of_model> --model_path <path_to_model_weights> --BUCC_folder <path_to_BUCC_data> --FLORES_folder <path_to_FLORES_data> --eval_folder <path_to_eval_folder>
python evaluation/evaluate.py --model labse --BUCC_folder <path_to_BUCC_data> --FLORES_folder <path_to_FLORES_data> --eval_folder <path_to_eval_folder>
python evaluation/evaluate.py --model word_emb --BUCC_folder <path_to_BUCC_data> --FLORES_folder <path_to_FLORES_data> --eval_folder <path_to_eval_folder>
```

### Inference
