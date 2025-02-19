from typing import Tuple

import Levenshtein
import pandas as pd


def count_unk_tokens(df: pd.DataFrame, column_name: str) -> int:
    """
    Count the number of <unk> tokens in the specified column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentences.
        column_name (str): The name of the column containing the sentences.

    Returns:
        int: The total number of <unk> tokens.
    """
    # Count the occurrences of <unk> in each sentence
    df["unk_count"] = df[column_name].str.count(r"<unk>")
    total_unk_tokens = df["unk_count"].sum()

    return total_unk_tokens


def count_sentences_in_bins(
    df: pd.DataFrame, metric: str, B1: float = 0.25, B2: float = 0.50, B3: float = 0.75
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Counts the number of sentences falling into different bins based on a given metric.

    Args:
        df (pd.DataFrame): The DataFrame containing the sentences and their associated metric.
        metric (str): The column name representing the metric based on which sentences will be categorized.
        B1 (float, optional): The threshold for the lower score, default is 0.25.
        B2 (float, optional): The threshold for the second bin, default is 0.50.
        B3 (float, optional): The threshold for the third bin, default is 0.75.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains the original DataFrame with an additional column indicating the bin each sentence belongs to.
            - The second DataFrame contains the counts of sentences in each bin.

    Example:
        Consider a DataFrame 'df' with columns ['sentence', 'bleu_score']. To count sentences in bins based on 'bleu_score', one can use:
        >>> df, bin_counts = count_sentences_in_bins(df, 'bleu_score')
    """

    df = df.sort_values(by=metric).reset_index(drop=True)

    # categorize each metric
    def categorize_quartile(bleu: float) -> str:
        if bleu <= B1:
            return "poor"
        elif bleu <= B2:
            return "low"
        elif bleu <= B3:
            return "moderate"
        else:
            return "high"

    df["bin"] = df[metric].apply(categorize_quartile)

    # Count the number of sentences in each bin
    bin_counts = df["bin"].value_counts().sort_index()
    bin_counts = bin_counts.to_frame().reset_index()  # convert into dataframe
    bin_counts.columns = ["bin", "count"]

    # Rearrange the index
    desired_index_order = ["poor", "low", "moderate", "high"]
    bin_counts = bin_counts.set_index("bin").reindex(desired_index_order).reset_index()

    return df, bin_counts


def average_length_by_bin(
    df: pd.DataFrame, sentence_column: str, bin_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the average length of sentences in a DataFrame grouped by a specified bin column.

    Args:
        df (pd.DataFrame): The input DataFrame containing sentences.
        sentence_column (str): The name of the column in the DataFrame that contains sentences.
        bin_column (str): The name of the column to group by for calculating the average sentence length.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The original DataFrame with an additional column for sentence lengths.
            - A DataFrame with the average sentence length for each bin, rounded to the nearest integer.
    """
    # Calculate the length of each sentence
    sent_len = lambda sentence: len(sentence.split())
    df[f"{sentence_column}_len"] = df[sentence_column].apply(sent_len)

    # Group by 'bin' and calculate the average length
    avg_len = df.groupby(bin_column)[f"{sentence_column}_len"].mean().reset_index()
    rnd_val = lambda val: round(val)
    avg_len[f"{sentence_column}_len"] = avg_len[f"{sentence_column}_len"].apply(rnd_val)

    return df, avg_len


def find_closest_translations(
    word: str,
    dictionaries: pd.DataFrame,
    top_n: int = 3,
    source: str = "indonesia",
    target: str = "wolio",
) -> pd.DataFrame:
    """
    Find the closest translations for a given word based on Levenshtein distance.

    Args:
        word (str): The word to find translations for.
        dictionaries (pd.DataFrame): A DataFrame containing source and target language columns.
        top_n (int, optional): The number of closest matches to return. Defaults to 3.
                               Set to 0 to return all matches.
        source (str, optional): The source language column name in the DataFrame.
        target (str, optional): The target language column name in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the closest translations with their distances.
    """

    # Calculate Levenshtein distances
    distances = dictionaries[source].apply(lambda x: Levenshtein.distance(word, x))

    distance_df = pd.DataFrame(
        {
            f"{source}": dictionaries[source],
            f"{target}": dictionaries[target],
            "distance": distances,
        }
    )

    # get the top N matches
    if top_n > 0:
        top_matches = distance_df.sort_values("distance").head(top_n)
    else:
        top_matches = distance_df.sort_values("distance")

    return top_matches


def word_to_word_translation(
    sentence: str,
    dictionaries: pd.DataFrame,
    source: str = "indonesia",
    target: str = "wolio",
    unknown: str = "<unk>",
) -> str:
    """
    Translate a sentence word by word using a dictionary DataFrame.

    Args:
        sentence (str): The sentence to translate.
        dictionaries (pd.DataFrame): A DataFrame containing source and target language columns.
        source (str, optional): The source language column name in the DataFrame.
        target (str, optional): The target language column name in the DataFrame.
        unknown (str, optional): The string to use for unknown words.

    Returns:
        str: The translated sentence.
    """

    words = sentence.split()
    translated_words = []

    for word in words:
        if word in dictionaries[source].values:
            translated_word = dictionaries.loc[
                dictionaries[source] == word, target
            ].values[0]
        else:
            translated_word = unknown
        translated_words.append(translated_word)

    return " ".join(translated_words)
