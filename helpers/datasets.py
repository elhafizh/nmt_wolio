from typing import Dict, List, Sequence, Tuple

import pandas as pd


def download_wolio_dataset(link: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sentences, vocabularies = pd.read_excel(
        link, sheet_name="Sentence Pair"
    ), pd.read_excel(link, sheet_name="Dictionary")
    return sentences, vocabularies
