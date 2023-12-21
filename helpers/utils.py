import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from nltk.util import ngrams
from torch import Tensor

from helpers import f_regex


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


def write_mtdata_to_files(
    df,
    source_file,
    target_file,
    suffix,
    encoding="utf-8",
):
    """
    Write a DataFrame to source and target files with the specified suffix.

    Args:
        df (pd.DataFrame): The DataFrame to be written to files.
        source_file (str): File path for the source language data.
        target_file (str): File path for the target language data.
        suffix (str): Suffix to be added to the file names.
        encoding (str, optional): Encoding for writing files. Default is 'utf-8'.
    """
    source_path = Path(source_file)
    target_path = Path(target_file)

    create_folder_if_not_exists("dataset/split")
    saved_dir = str(source_path.parent) + "/split/"

    source_filename = source_path.name + suffix
    target_filename = target_path.name + suffix
    source_filename = saved_dir + source_filename
    target_filename = saved_dir + target_filename
    df_dict = df.to_dict(orient="list")

    with open(source_filename, "w", encoding=encoding) as sf:
        sf.write("\n".join(line for line in df_dict["Source"]))
        sf.write("\n")  # end of file newline

    with open(target_filename, "w", encoding=encoding) as tf:
        tf.write("\n".join(line for line in df_dict["Target"]))
        tf.write("\n")  # end of file newline

    return source_filename, target_filename


def write_to_file(filename: str, content: str) -> None:
    """
    Write content to a file with the specified filename.

    Args:
        filename (str): The name of the file to write.
        content (str): The content to be written to the file.

    Example:
        write_to_file("example.txt", "This is the content to be written to the file.")
    """
    with open(filename, "w+") as file:
        file.write(content.expandtabs(4))


def copy_files(source_folder: str, target_folder: str) -> None:
    """
    Copies all files from the source folder to the target folder.

    Args:
        source_folder (str): The path to the source folder containing files to be copied.
        target_folder (str): The path to the target folder where files will be copied to.
    """
    # Create the target directory if it doesn't exist
    create_folder_if_not_exists(target_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    for file in files:
        if file != "data.csv":
            source_path = os.path.join(source_folder, file)
            target_path = os.path.join(target_folder, file)
            try:
                shutil.copy2(source_path, target_path)
            except IsADirectoryError as err:
                if not f_regex.is_hidden(file):
                    command = [
                        "cp",
                        "-r",
                        f"{source_folder}/{file}/",
                        f"{target_folder}/",
                    ]
                    create_folder_if_not_exists(f"{target_folder}/{file}")
                    execute_cmd(command)


def execute_cmd(command: List[str], log_output: bool = False) -> str:
    """
    Execute a command in the system shell.

    Args:
        command (List[str]): A list containing the command and its arguments.
        log_output (bool, optional): If True, save the output to a file. Default is False.

    Returns:
        str: The standard output of the command if 'log_output' is False, an empty string otherwise.

    Example:
        >>> execute_cmd(["ls", "-l"])
        'total 8\n-rw-rw-r-- 1 user user  12 Dec  1 12:00 example.txt\n'

        >>> execute_cmd(["ls", "-l"], log_output=True)
        # Executes the command and saves the output to a file with a generated filename.

    Note:
        If 'log_output' is True, the standard output is saved to a file with a filename
        generated using 'generate_log_filename()' and the command's name.

    """
    if not log_output:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            print("Error executing command")
            print("Error:", result.stderr)
            return ""
    else:
        with open(f"{generate_log_filename()}_{command[0]}.txt", "w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.PIPE, text=True)


def get_cpu_count() -> int:
    """Get the number of CPUs/cores on the machine using the 'nproc' command.

    Returns:
        int: The number of CPUs/cores on the machine.

    Example:
        >>> get_cpu_count()
        4
    """
    commands = ["nproc", "--all"]
    num = int(execute_cmd(commands))

    return num


def generate_log_filename() -> str:
    """Generate a timestamped log file name in the format 'YYYYMMDD_HHMMSS'.

    Returns:
        str: The timestamped log file name.

    Note:
        This function uses the current date and time to create a timestamped log
        file name in a human-readable and sortable format. The format is
        'YYYYMMDD_HHMMSS', where 'YYYY' is the year, 'MM' is the month,
        'DD' is the day, 'HH' is the hour, 'MM' is the minute, and 'SS' is the
        second.

    Example:
        >>> generate_log_filename()
        '2023_12_08_153021'
    """
    log_filename = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return log_filename


def check_gpu_info():
    """
    Check GPU availability, name, and free memory.

    This function prints information about the availability of GPU,
    the name of the GPU (if available), and the free GPU memory.

    Example:
        >>> check_gpu_info()
        GPU is available.
        GPU Name: NVIDIA GeForce GTX 1080
        Free GPU memory: 8000.0 MB out of: 8192.0 MB
    """
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("GPU is available.")
        # Get the name of the GPU
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU Name:", gpu_name)

        # Get GPU memory information
        gpu_memory = torch.cuda.mem_get_info(0)
        free_memory_mb = gpu_memory[0] / 1024**2
        total_memory_mb = gpu_memory[1] / 1024**2

        print("Free GPU memory:", free_memory_mb, "MB out of:", total_memory_mb, "MB")
    else:
        print("GPU is not available.")


def create_new_file(file_name: str, content: str = ""):
    """
    Create a new file and write content to it.

    Args:
        file_name (str): The name of the new file.
        content (str): The content to be written to the file.
    """
    with open(file_name, "w") as file:
        file.write(content)
    print(f"File '{file_name}' created successfully.")


def load_eval_set(target_test: str, target_pred: str):
    """Load evaluation set for machine translation.

    Args:
        target_test (str): The path to the detokenized file containing human translations.
        target_pred (str): The path to the detokenized file containing machine translations.

    Returns:
        tuple: A tuple containing two elements:
            - refs (List[List[str]]): A list of reference translations. Each reference is a list of strings.
            - preds (List[str]): A list of machine translations.

    Note:
        Make sure that the target_test and target_pred files are already detokenized.

    Example:
        refs, preds = load_eval_set("human_translations.txt", "machine_translations.txt")
    """
    refs = []

    with open(target_test) as test:
        for line in test:
            line = line.strip()
            # the metric evaluation required list of list
            refs.append([line])

    preds = []

    with open(target_pred) as pred:
        for line in pred:
            line = line.strip()
            preds.append(line)

    return refs, preds


def save_dataframe_to_csv(
    dataframe: pd.DataFrame,
    filename: str,
    include_index: bool = False,
):
    """
    Save a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        filename (str): The name of the CSV file (including extension).
        include_index (bool, optional): Whether to include the index in the CSV file. Default is False.

    Example:
        data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
                'Age': [25, 30, 22, 35],
                'Salary': [50000, 60000, 45000, 70000]}

        df = pd.DataFrame(data)

        # Save DataFrame to a CSV file without index
        save_dataframe_to_csv(df, 'output.csv')

        # Save DataFrame to a CSV file with index
        # save_dataframe_to_csv(df, 'output_with_index.csv', include_index=True)
    """
    create_folder_if_not_exists(str(Path(filename).parent))
    dataframe.to_csv(filename, index=include_index)
