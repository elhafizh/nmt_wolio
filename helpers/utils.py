import os
import shutil
from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import torch
from mpl_toolkits.mplot3d import Axes3D
from nltk.util import ngrams
from torch import Tensor


def visualize_3Dtensor_inscatter(tensor_3d: Tensor) -> None:
    x = tensor_3d[:, :, 0].flatten()
    y = tensor_3d[:, :, 1].flatten()
    z = tensor_3d[:, :, 2].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    plt.title("3D Scatter Plot of the Tensor")
    plt.show()


def char_ngram(sentence: str, n: int = 1) -> List[Tuple[str, str]]:
    sentence = "".join(sentence.split()).lower()
    sentence = [char for char in sentence]
    if n == 1:
        sentence = list(ngrams(sentence, n))
    elif n == 2:
        sentence = list(ngrams(sentence, n))
    else:
        print("undefined ngram")
        return
    return sentence


def char_matching_ngram(ref: str, hyp: str, n: int = 1) -> List[Tuple[str, str]]:
    if n <= 2:
        ref = char_ngram(ref, n)
        hyp = char_ngram(hyp, n)
        matching_ngram = [gram for gram in ref if gram in hyp]
        return matching_ngram
    else:
        print("undefined ngram")
        return


def create_folder_if_not_exists(folder_path: str):
    """
    Verify if a folder exists. If not, create a new one.

    Args:
        folder_path (str): The path of the folder to be verified/created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' exists.")


def move_file(source_path: str, destination_path: str):
    """
    Move a file from the source path to the destination path.

    Args:
        source_path (str): The path of the file to be moved.
        destination_path (str): The destination path where the file will be moved.

    Returns:
        None: The function moves the file to the specified destination.

    Raises:
        FileNotFoundError: If the source file does not exist.
    """
    try:
        shutil.move(source_path, destination_path)
        print(f"File moved successfully from {source_path} to {destination_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {source_path} does not exist.")


def get_filename_from_path(file_path: str):
    """
    Extract the filename from a given file path.

    Args:
        file_path (str): The full path of the file.

    Returns:
        str: The filename extracted from the file path.

    Raises:
        ValueError: If the provided path is a directory.
    """
    if os.path.isdir(file_path):
        raise ValueError("Provided path is a directory, not a file path.")

    return os.path.basename(file_path)
