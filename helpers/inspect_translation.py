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
