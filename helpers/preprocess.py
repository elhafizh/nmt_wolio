import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from tqdm.auto import tqdm


def samples_for_training_tokenizer(
    dataset_train: Sequence,
    dataset_name: str,
) -> str:
    text_data = []
    save_file_on = f"./datas/{dataset_name}/"
    if not os.path.exists(save_file_on):
        os.mkdir(save_file_on)
    if isinstance(dataset_train, pd.Series):
        text_data = dataset_train.tolist()
    else:
        for sample in tqdm(dataset_train):
            text_data.append(sample)
    with open(
        f"{save_file_on}/text_{dataset_name}.txt",
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write("\n".join(text_data))
    return f"Text Data for training tokenizer saved on {save_file_on}"


def train_tokenizer(
    path_samples: str,
    vocab_size: int = 30_522,
    min_frequency: int = 2,
) -> List:
    paths = [str(x) for x in Path(path_samples).glob("**/*.txt")]
    # print(paths)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    os.mkdir(f"{path_samples}/tokenizer")
    return tokenizer.save_model(f"{path_samples}/tokenizer")
