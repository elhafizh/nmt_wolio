from typing import Tuple

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
