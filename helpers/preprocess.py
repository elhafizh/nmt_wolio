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


import sentencepiece as spm

from . import utils


def train_sentencepiece(
    source_file_path: str,
    target_file_path: str,
    model_type: str = "bpe",
    vocab_limit: str = "false",
):
    """
    Train SentencePiece models for source and target languages using the specified input files.

    Args:
        source_file_path (str): The file path to the source language training data.
        target_file_path (str): The file path to the target language training data.
        model_type (str, optional): The type of SentencePiece model to train, e.g., "bpe" (Byte Pair Encoding).
                                    Default is "bpe".
        vocab_limit (str, optional): If needed due to training data being too small, set to False.

    Returns:
        None: The function saves the trained models as 'source.model', 'target.model',
              and their respective vocabularies as 'source.vocab', 'target.vocab'.

    Notes:
        - The training data is tokenized into subword pieces using SentencePiece.
        - The vocabulary size for each model is set to 50000.
        - Digits are split into separate subword pieces with the --split_digits parameter set to True.
    """

    # Source subword model

    source_prefix = "tokens." + utils.get_filename_from_path(source_file_path)

    source_train_value = (
        "--input="
        + source_file_path
        + " --model_prefix="
        + source_prefix
        + " --vocab_size=50000 --hard_vocab_limit="
        + vocab_limit
        + " --model_type="
        + model_type
        + " --split_digits=true"
    )
    spm.SentencePieceTrainer.train(source_train_value)
    print("Training of SentencePiece model for the Source completed successfully!")

    # Target subword model

    target_prefix = "tokens." + utils.get_filename_from_path(target_file_path)

    target_train_value = (
        "--input="
        + target_file_path
        + " --model_prefix="
        + target_prefix
        + " --vocab_size=50000 --hard_vocab_limit="
        + vocab_limit
        + " --model_type="
        + model_type
        + " --split_digits=true"
    )
    spm.SentencePieceTrainer.train(target_train_value)
    print("Training of SentencePiece model for the Target completed successfully!")

    # Move the subword model into the dataset folder
    utils.create_folder_if_not_exists("./dataset")
    models_created = [
        f"{source_prefix}.model",
        f"{source_prefix}.vocab",
        f"{target_prefix}.model",
        f"{target_prefix}.vocab",
    ]
    for fl in models_created:
        utils.move_file(fl, "dataset/")
