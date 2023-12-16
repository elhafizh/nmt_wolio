from helpers import datasets

if __name__ == "__main__":
    # load wolio dataset
    wlo_ind_new_sentences, wlo_ind_new_dictionaries = datasets.download_wolio_dataset(
        "/content/KB_2017_Raw.xlsx"
    )
    wlo_ind_old_sentences, wlo_ind_old_dictionaries = datasets.download_wolio_dataset(
        "/content/KB_TD_Raw.xlsx"
    )
