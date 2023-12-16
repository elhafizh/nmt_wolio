import csv
import os
from pathlib import Path
from typing import Annotated, Dict, List, Sequence, Tuple, Union

import pandas as pd

from . import f_regex, utils


def load_mt_dataset(
    link: str,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame,]:
    """Fetches data from an external source.

    Args:
        link (str): The location of the dataset.

    Returns:
        Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
            A tuple containing two DataFrames if the dataset is in Excel format.
            The first DataFrame represents the 'Sentence Pair' sheet, and the second DataFrame represents
            the 'Dictionary' sheet. If the dataset is in another format (assumed to be CSV),
            a single DataFrame is returned with column names derived from the file name.
    """

    file_path = Path(link)

    # Compare the file extension (remove dot)
    if file_path.suffix[1:] == "xlsx":
        sentences, vocabularies = pd.read_excel(
            link, sheet_name="Sentence Pair"
        ), pd.read_excel(link, sheet_name="Dictionary")

        # Make all column names lowercase
        sentences.columns = sentences.columns.str.lower()
        vocabularies.columns = vocabularies.columns.str.lower()

        return sentences, vocabularies
    else:
        # Make the file name as column name
        column_name = file_path.stem + file_path.suffix

        dataset = pd.read_csv(file_path, sep="delimiter", names=[column_name])

        return dataset


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
    
    # Delete nan
    df_authentics_ind_wlo = df_authentics_ind_wlo.dropna()

    print("Dataset shape (rows, columns):", df_authentics_ind_wlo.shape)

    # remove unnecessary symbols
    df_authentics_ind_wlo.wolio = df_authentics_ind_wlo.wolio.apply(
        lambda x: f_regex.delete_istl_from_sentence(x)
    )
    df_authentics_ind_wlo.wolio = df_authentics_ind_wlo.wolio.apply(
        lambda x: f_regex.delete_words_from_pb(x)
    )
    df_authentics_ind_wlo.wolio = df_authentics_ind_wlo.wolio.apply(
        lambda x: f_regex.remove_sentence_after_asterisk(x)
    )

    # Save source and target to two text files
    df_source = df_authentics_ind_wlo.indonesia
    df_target = df_authentics_ind_wlo.wolio

    # Make the columns lowercase
    if is_lowercase:
        df_source = df_source.str.lower()
        df_target = df_target.str.lower()
        # Remove apostrophes from the 'sentence'
        df_source = df_source.str.replace("'", "")
        df_target = df_target.str.replace("'", "")

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
