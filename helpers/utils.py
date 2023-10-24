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
