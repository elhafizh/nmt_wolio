import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer
from tqdm.auto import tqdm

from . import utils


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


def sentence_subwording(
    source_model: str,
    target_model: str,
    source_raw: str,
    target_raw: str,
):
    """
    Subword tokenize the source and target datasets using SentencePiece models.

    Args:
        source_model (str): Path to the source language SentencePiece model.
        target_model (str): Path to the target language SentencePiece model.
        source_raw (str): Path to the raw source language dataset.
        target_raw (str): Path to the raw target language dataset.

    Returns:
        str, str: Paths to the subword-tokenized source and target datasets.
    """
    source_subworded = source_raw + ".subword"
    target_subworded = target_raw + ".subword"

    print("Source Model:", source_model)
    print("Target Model:", target_model)
    print("Source Dataset:", source_raw)
    print("Target Dataset:", target_raw)

    sp = spm.SentencePieceProcessor()

    # Subwording the train source
    sp.load(source_model)
    with open(source_raw, encoding="utf-8") as source, open(
        source_subworded, "w+", encoding="utf-8"
    ) as source_subword:
        for line in source:
            line = line.strip()
            line = sp.encode_as_pieces(line)
            # line = ['<s>'] + line + ['</s>']    # add start & end tokens; optional
            line = " ".join([token for token in line])
            source_subword.write(line + "\n")

    print("Done subwording the source dataset! Output:", source_subworded)

    # Subwording the train target
    sp.load(target_model)
    with open(target_raw, encoding="utf-8") as target, open(
        target_subworded, "w+", encoding="utf-8"
    ) as target_subword:
        for line in target:
            line = line.strip()
            line = sp.encode_as_pieces(line)
            # line = ['<s>'] + line + ['</s>']    # add start & end tokens; unrequired for OpenNMT
            line = " ".join([token for token in line])
            target_subword.write(line + "\n")

    print("Done subwording the target dataset! Output:", target_subworded)

    return source_subworded, target_subworded


def sentence_desubword(target_model: str, target_pred: str):
    """Desubword sentences using a SentencePiece model.

    Args:
        target_model (str): Path to the SentencePiece model file.
        target_pred (str): Path to the file containing subword-encoded sentences for desubwording.
    """
    target_decodeded = target_pred + ".desubword"

    sp = spm.SentencePieceProcessor()
    sp.load(target_model)

    with open(target_pred) as pred, open(target_decodeded, "w+") as pred_decoded:
        for line in pred:
            line = line.strip().split(" ")
            line = sp.decode_pieces(line)
            pred_decoded.write(line + "\n")

    print("Done desubwording! Output:", target_decodeded)
    return target_decodeded


def split_dataset_segment(num_dev, num_test, source_file, target_file):
    """
    Split a parallel dataset into training, development, and test sets.

    Args:
        num_dev (int): Number of samples to include in the development set.
        num_test (int): Number of samples to include in the test set.
        source_file (str): File path for the subword-tokenized source language data.
        target_file (str): File path for the subword-tokenized target language data.

    Returns:
        None

    This function reads parallel text data from source and target files,
    combines them into a dataframe, and then splits the data into training,
    development, and test sets. The resulting sets are written to separate files
    for both source and target languages.

    Files created:
        source_file.train, target_file.train: Training set files
        source_file.dev, target_file.dev: Development set files
        source_file.test, target_file.test: Test set files
    """

    # Read data from source and target files
    df_source = pd.read_csv(
        source_file,
        names=["Source"],
        sep="\0",
        quoting=csv.QUOTE_NONE,
        skip_blank_lines=False,
        on_bad_lines="skip",
    )
    df_target = pd.read_csv(
        target_file,
        names=["Target"],
        sep="\0",
        quoting=csv.QUOTE_NONE,
        skip_blank_lines=False,
        on_bad_lines="skip",
    )
    df = pd.concat(
        [df_source, df_target], axis=1
    )  # Join the two dataframes along columns

    # Delete rows with empty cells (source or target)
    df = df.dropna()

    # Extract Test set from the main dataset
    df_test = df.sample(n=int(num_test))
    df_train = df.drop(df_test.index)

    # Extract Dev set
    df_dev = df_train.sample(n=int(num_dev))

    """Write the dataframe to two Source and Target files"""

    # training set
    source_file_train, target_file_train = utils.write_mtdata_to_files(
        df_train,
        source_file,
        target_file,
        ".train",
    )

    # development set
    source_file_dev, target_file_dev = utils.write_mtdata_to_files(
        df_dev,
        source_file,
        target_file,
        ".dev",
    )

    # test set
    source_file_test, target_file_test = utils.write_mtdata_to_files(
        df_test,
        source_file,
        target_file,
        ".test",
    )

    print(
        "Output files",
        *[
            source_file_train,
            target_file_train,
            source_file_dev,
            target_file_dev,
            source_file_test,
            target_file_test,
        ],
        sep="\n",
    )
