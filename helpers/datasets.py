import csv
import os
from typing import Annotated, Dict, List, Sequence, Tuple

import pandas as pd

from . import f_regex, utils


def load_mt_dataset(link: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetching data from external source

    Args:
        link: dataset location

    Returns:
        tuple: A tuple of dataframe which consists of parallel sentences
            and a reference dictionary.
    """
    sentences, vocabularies = pd.read_excel(
        link, sheet_name="Sentence Pair"
    ), pd.read_excel(link, sheet_name="Dictionary")

    # Make all column names lowercase
    sentences.columns = sentences.columns.str.lower()
    vocabularies.columns = vocabularies.columns.str.lower()

    return sentences, vocabularies


def prepare_authentic_dataset(
    paths: List,
    is_lowercase: bool = False,
) -> pd.DataFrame:
    """Shuffle and write dataframe to local files

    Args:
        paths (List): A list of strings representing file locations.
        is_lowercase (bool, optional):
            If True, converts the strings in the resulting DataFrame to lowercase

    Returns:
        pd.DataFrame: a dataframe of parallel sentences
    """
    df_authentics = []

    for path in paths:
        df_authentics.append(load_mt_dataset(path))

    df_authentics_ind_wlo = pd.concat(
        [df_authentics[0][0], df_authentics[1][0]], ignore_index=True
    )

    df_authentics_ind_wlo = df_authentics_ind_wlo.sample(frac=1).reset_index(
        drop=True,
    )

    print("Dataset shape (rows, columns):", df_authentics_ind_wlo.shape)

    # remove unnecessary symbols
    df_authentics_ind_wlo.wolio = df_authentics_ind_wlo.wolio.apply(
        lambda x: f_regex.delete_istl_from_sentence(x)
    )
    df_authentics_ind_wlo.wolio = df_authentics_ind_wlo.wolio.apply(
        lambda x: f_regex.delete_words_from_pb(x)
    )

    # Save source and target to two text files
    df_source = df_authentics_ind_wlo.indonesia
    df_target = df_authentics_ind_wlo.wolio

    # Make the columns lowercase
    if is_lowercase:
        df_source = df_source.str.lower()
        df_target = df_target.str.lower()

    utils.create_folder_if_not_exists("./dataset")

    df_source.to_csv(
        "dataset/authentic.ind",
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        sep="\n",
    )
    df_target.to_csv(
        "dataset/authentic.wlo",
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        sep="\n",
    )

    return df_authentics_ind_wlo
