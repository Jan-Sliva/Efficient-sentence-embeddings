import tensorflow_datasets as tfds
import os

dir = "./tatoeba"
os.makedirs(dir, exist_ok=True)

dataset_configs = tfds.builder('tatoeba').builder_configs

for dataset_name in dataset_configs:
    dataset = tfds.load(f"tatoeba/{dataset_name}", split='train')

    output_dir = f"{dir}/{dataset_name[-2:]}"
    os.makedirs(output_dir, exist_ok=True)

    source_file = f"{output_dir}/source.txt"
    target_file = f"{output_dir}/target.txt"

    # Extract sentences and write to file
    with open(source_file, 'w', encoding='utf-8') as source_f:
        with open(target_file, 'w', encoding='utf-8') as target_f:
            for example in dataset:
                # Decode bytes to string and remove newlines
                source = example['source_sentence'].numpy().decode('utf-8').strip()
                target = example['target_sentence'].numpy().decode('utf-8').strip()
                source_f.write(f"{source}\n")
                target_f.write(f"{target}\n")
