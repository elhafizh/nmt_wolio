import os
from dataclasses import InitVar, dataclass, is_dataclass
from pathlib import Path
from typing import List, Tuple, Type, Union

import pandas as pd
from torchmetrics.text import CHRFScore, SacreBLEUScore, TranslationEditRate
from tqdm.auto import tqdm, trange

from helpers import f_regex, preprocess, utils


@dataclass
class TransformerEssential:
    """
    Configuration class for defining essential parameters of a Transformer model.

    Attributes:
        n_enc (int): Number of encoder stacks.
        n_dec (int): Number of decoder stacks.
        d_ff (int): Hidden units in fully connected layers.
        h (int): Number of heads in multi-head attention layers.
        p_drop (float): Residual Dropout probability.
        e_ls (float): Label Smoothing factor.
        d_model (int): Word embedding size for both source (src) and target (tgt) sequences.
        model_type (str, optional): Type of the Transformer model. Defaults to "transformer".
        position_encoding (bool, optional): Type of position encoding. Defaults to True.
    """

    n_enc: int
    n_dec: int
    d_ff: int
    h: int
    p_drop: float
    e_ls: float
    d_model: int
    model_type: str = "transformer"
    position_encoding: bool = True

    def __post_init__(self):
        config = f"\n## Transformer Essential Parameters \n\n"
        config = config + f"encoder_type: {self.model_type} \n"
        config = config + f"decoder_type: {self.model_type} \n"
        config = config + f"enc_layers: {self.n_enc} \n"
        config = config + f"dec_layers: {self.n_dec} \n"
        config = config + f"transformer_ff: {self.d_ff} \n"
        config = config + f"heads: {self.h} \n"
        config = config + f"position_encoding: {bool_yaml(self.position_encoding)} \n"
        config = config + f"dropout: {self.p_drop} \n"
        config = config + f"label_smoothing: {self.e_ls} \n"
        config = config + f"word_vec_size: {self.d_model} \n "
        self.config = config


@dataclass
class TrainingFlow:
    """
    Configuration class for dictating the training flow.

    Attributes:
        train_steps (int): Maximum number of training steps.
        valid_steps (int): Interval for running validation during training.
        warmup_steps (int): Number of warm-up steps for adjusting learning rates.
        batch_size (int): Maximum batch size for training.
        report_every (int): Interval for printing statistics during training.
        save_checkpoint (int): Interval for saving model checkpoints.
        keep_checkpoint (int): Limit on the number of recent checkpoints to be saved for space efficiency.
        accum_count (int): Number of times to accumulate gradients before updating the model weights.
    """

    train_steps: int
    valid_steps: int
    warmup_steps: int
    batch_size: int
    report_every: int
    save_checkpoint: int
    keep_checkpoint: int
    accum_count: int

    def __post_init__(self):
        # Maximum batch size for validation. Default: 32
        valid_batch_size = round(self.batch_size / 2)
        config = f"\n \n## Training Flow \n \n"
        config = config + f"# Train the model to max n steps\n"
        config = config + f"train_steps: {self.train_steps}\n"
        config = config + f"batch_size: {self.batch_size}\n"
        config = config + f"# Run validation after n steps\n"
        config = config + f"valid_steps: {self.valid_steps}\n"
        config = config + f"valid_batch_size: {valid_batch_size}\n"
        config = config + f"warmup_steps: {self.warmup_steps}\n"
        config = config + f"report_every: {self.report_every}\n"
        config = config + f"save_checkpoint_steps: {self.save_checkpoint}\n"
        config = config + f"keep_checkpoint: {self.keep_checkpoint}\n"
        config = config + f"accum_count: {self.accum_count} \n "
        self.config = config


@dataclass
class TrainingComplementary:
    """
    Containing supplementary attributes for training flow.
    All attributes follow default recommendation value.

    Attributes:
        optim (str, optional): Optimization method. Defaults to "adam".
        adam_beta2 (float, optional): The beta2 parameter used by Adam. Defaults to 0.998.
        decay_method (str, optional): Custom decay rate method. Defaults to "noam".
        learning_rate (float, optional): Starting learning rate. Defaults to 2.0.
        max_grad_norm (float, optional): If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm. Defaults to 0.0.
        batch_type (str, optional): Batch grouping method for batch size. Defaults to "tokens".
        normalization (str, optional): Normalization method of the gradient. Defaults to "tokens".
        param_init (float, optional): Parameters initialization over a uniform distribution with support. Use 0 to not use initialization. Defaults to 0.
        param_init_glorot (bool, optional): Initialize parameters with xavier_uniform. Required for transformers. Defaults to True.
        world_size (int, optional): Total number of distributed processes. Defaults to 1.
        hidden_size (int, optional): Overwrites enc_hid_size and dec_hid_size. Defaults to 512.
    """

    optim: str = "adam"
    adam_beta2: float = 0.998
    decay_method: str = "noam"
    learning_rate: float = 2.0
    max_grad_norm: float = 0.0
    batch_type: str = "tokens"
    normalization: str = "tokens"
    param_init: float = 0
    param_init_glorot: bool = True
    world_size: int = 1
    hidden_size: int = 512

    def __post_init__(self):
        gpu_ranks = f"\n# Number of GPUs, and IDs of GPUs\n"
        gpu_ranks = f"world_size: {self.world_size}\n"
        gpu_ranks = gpu_ranks + f"gpu_ranks:\n"
        for i in range(self.world_size):
            gpu_ranks = gpu_ranks + f"- {i}\n"

        config = f"\n \n## Training Complementary \n \n"
        config = config + f"optim: {self.optim}\n"
        config = config + f"adam_beta2: {self.adam_beta2}\n"
        config = config + f"decay_method: {self.decay_method}\n"
        config = config + f"learning_rate: {self.learning_rate}\n"
        config = config + f"max_grad_norm: {self.max_grad_norm}\n"
        config = config + f"batch_type: {self.batch_type}\n"
        config = config + f"normalization: {self.normalization}\n"
        config = config + f"param_init: {self.param_init}\n"
        config = config + f"param_init_glorot: {bool_yaml(self.param_init_glorot)}\n"
        config = config + f"hidden_size: {self.hidden_size}\n"
        config = config + gpu_ranks
        self.config = config


@dataclass
class TrainingExtra:
    """
    Defining additional data archetype.

    Attributes:
        src_vocab_size (int, optional): Vocabulary size for the source language, should be the same as in SentencePiece. Defaults to 50000.
        tgt_vocab_size (int, optional): Vocabulary size for the target language, should be the same as in SentencePiece. Defaults to 50000.
        src_seq_length (int, optional): Maximum source sequence length. Defaults to 150.
        tgt_seq_length (int, optional): Maximum target sequence length. Defaults to 150.
        early_stopping (int, optional): Stop training if it does not improve after n validations. Defaults to 4.
        num_workers (int, optional): Number of worker processes for data loading. Set to 0 when running out of RAM. Defaults to 0.
        model_dtype (str, optional): Data type of the model. Defaults to "fp16".
    """

    src_vocab_size: int = 50000
    tgt_vocab_size: int = 50000
    src_seq_length: int = 150
    tgt_seq_length: int = 150
    early_stopping: int = 4
    num_workers: int = 0
    model_dtype: str = "fp16"

    def __post_init__(self):
        config = f"\n \n# Additional Config \n \n"
        config = config + f"src_vocab_size: {self.src_vocab_size}\n"
        config = config + f"tgt_vocab_size: {self.tgt_vocab_size}\n"
        config = config + f"src_seq_length: {self.src_seq_length}\n"
        config = config + f"tgt_seq_length: {self.tgt_seq_length}\n"
        config = config + f"early_stopping: {self.early_stopping}\n"
        config = config + f"num_workers: {self.num_workers}\n"
        config = config + f'model_dtype: "{self.model_dtype}"\n'
        self.config = config


@dataclass
class TrainingRecord:
    """
    To store essential paths, filenames, and naming conventions for saving models

    Attributes:
        root_dir (InitVar[str]): Directory to store all training-related data.
        save_data (InitVar[str]): Output base path for objects that will be saved (vocab, transforms, embeddings, etc.).
        path_src_train (str): Location of the training source files.
        path_tgt_train (str): Location of the training target files.
        path_src_valid (str): Location of the validation source files.
        path_tgt_valid (str): Location of the validation target files.
        src_subword_model (str): Path of the subword model generated by SentencePiece for the source language.
        tgt_subword_model (str): Path of the subword model generated by SentencePiece for the target language.
        model_filename (str): The model will be saved as <save_model>_N.pt where N is the number of steps.
    """

    root_dir: InitVar[str]
    save_data: InitVar[str]
    path_src_train: str
    path_tgt_train: str
    path_src_valid: str
    path_tgt_valid: str
    src_subword_model: str
    tgt_subword_model: str
    model_filename: str

    def __post_init__(self, root_dir, save_data):
        utils.create_folder_if_not_exists(root_dir)
        self.main_dir = f"{root_dir}/{save_data}"

        # Vocabulary files, generated by onmt_build_vocab
        src_vocab: str = f"{self.main_dir}/source.vocab"
        tgt_vocab: str = f"{self.main_dir}/target.vocab"

        # structure the configuration
        config = f"\n## Where should the samples be recorded\n"
        config = config + f"save_data: {self.main_dir}\n\n"

        config = config + f"# Training files\n"
        config = config + f"data:\n\t"
        config = config + f"corpus1:\n\t\t"
        config = config + f"path_src: {self.path_src_train}\n\t\t"
        config = config + f"path_tgt: {self.path_tgt_train}\n\t\t"
        config = config + f"transforms: [filtertoolong]\n\t"
        config = config + f"valid:\n\t\t"
        config = config + f"path_src: {self.path_src_valid}\n\t\t"
        config = config + f"path_tgt: {self.path_tgt_valid}\n\t\t"
        config = config + f"transforms: [filtertoolong]\n\n"

        config = config + f"# Vocabulary files, generated by onmt_build_vocab\n"
        config = config + f"src_vocab: {src_vocab}\n"
        config = config + f"tgt_vocab: {tgt_vocab}\n\n"

        config = config + f"# Tokenization options\n"
        config = config + f"src_subword_model: {self.src_subword_model}\n"
        config = config + f"tgt_subword_model: {self.tgt_subword_model}\n\n"
        config = (
            config + f"# Where to save the log file and the output models/checkpoints\n"
        )
        utils.create_folder_if_not_exists(f"{self.main_dir}/models/")
        config = config + f"log_file: {self.main_dir}/models/train.log\n"
        utils.create_new_file(f"{self.main_dir}/models/train.log")
        utils.create_new_file(f"{self.main_dir}/models/train_tee.log")
        config = (
            config + f"save_model: {self.main_dir}/models/{self.model_filename}\n\n"
        )

        config = config + f"# Activating TensorBoard\n"
        config = config + f"tensorboard: true\n"
        self.tensorboard_log = f"{self.main_dir}/models/tensorboard"
        utils.create_folder_if_not_exists(f"{self.tensorboard_log}")
        config = config + f"tensorboard_log_dir: {self.tensorboard_log} \n\n"
        self.config = config.expandtabs(4)


def generateTrainingConfig(*args: Type[dataclass]) -> str:
    """
    Generate a training configuration by concatenating the `config` attribute of data class types
    and return the configuration content as a string.

    Args:
        *args (Type[dataclass]): Variable number of data class types.

    Returns:
        str: Concatenated configuration content.

    Raises:
        TypeError: If any argument is not a valid data class type.

    Example:
        config_content = generateTrainingConfig(DataClass1, DataClass2)
    """
    content = ""
    for arg in args:
        if not is_dataclass(arg):
            raise TypeError(
                f"Invalid argument type. Expected a data class type, but received: {type(arg)}"
            )

        content += arg.config

    return content


def build_vocabulary(config_file: str) -> None:
    """Build the vocabulary based on the specified OpenNMT configuration file.

    Args:
        config_file (str): The file path to the OpenNMT configuration file.

    Returns:
        None: The function does not return a value.

    Note:
        This function uses the 'onmt_build_vocab' command to build the vocabulary
        based on the provided OpenNMT configuration file. The vocabulary is
        created with a specified number of threads, utilizing the
        'utils.get_cpu_count()' function.

        The resulting vocabulary is stored in the './compilation' folder.

    Example:
        >>> build_vocabulary("my_config.yaml")
        # Executes the 'onmt_build_vocab' command based on the 'my_config.yaml'
        # file and creates the vocabulary in the './compilation' folder.
    """
    commands = [
        "onmt_build_vocab",
        "-config",
        config_file,
        "-n_sample",
        "-1",
        "-num_threads",
        f"{utils.get_cpu_count()}",
    ]
    utils.create_folder_if_not_exists("./compilation")
    utils.execute_cmd(commands)


def training(config_file: str, saved_log: str) -> None:
    """Train a model based on the specified OpenNMT configuration file.

    Args:
        config_file (str): The file path to the OpenNMT configuration file.
        saved_log (str): path to saved training log.

    Note:
        This function uses the 'onmt_train' command to initiate the training
        process for a neural machine translation model. The training is
        configured using the provided OpenNMT configuration file.

    """
    training_command = f"\
        onmt_train -config {config_file} |& tee {saved_log}"
    command = ["bash", "-c", training_command]
    utils.execute_cmd(command)


@dataclass
class TranslateEssential:
    """TranslateEssential is a data class for setting up OpenNMT translation configuration.

    Args:
        model (Union[str, List[str]]): Path to model .pt file(s). Multiple models can be specified
            for ensemble decoding.
        src (str): Path to the source sequence file to decode (one line per sequence).
        translated_by (str, optional): Specify model suffix to the translated file. Defaults to "".
        min_length (int, optional): Minimum prediction length. Defaults to 1.
        verbose (bool, optional): If True, print scores and predictions for each sentence.
            Defaults to False.
        tgt (str, optional): Path to the true target sequence file (optional). Defaults to "".
        replace_unk (bool, optional): If True, replace the generated UNK tokens with the source
            token that had the highest attention weight. Defaults to False.
        phrase_table (str, optional): If provided (with replace_unk), look up the identified
            source token in the phrase_table and give the corresponding target token. Defaults to "".
        enable_gpu (bool, optional): If True, enable GPU acceleration. Default is False.
    """

    model: Union[str, List[str]]
    src: str
    translated_by: str = ""
    min_length: int = 1
    verbose: bool = False
    tgt: str = ""
    replace_unk: bool = False
    phrase_table: str = ""
    enable_gpu: bool = False

    def __post_init__(self):
        config = f"\n## Data \n \n"
        if isinstance(self.model, list):
            config = config + f"model:\n"
            path_model = Path(self.model[0])
            for i in range(len(self.model)):
                config = config + f"- {self.model[i]}\n"
        elif isinstance(self.model, str):
            path_model = Path(self.model)
            config = config + f"model: {self.model}\n"
        model_type = str(path_model.parent.parent.name)
        config = config + f"src: {self.src}\n"
        path_src = Path(self.src)
        saved_dir = str(path_src.parent.parent)
        saved_dir = f"{saved_dir}/translated/{model_type}"
        utils.create_folder_if_not_exists(saved_dir)
        saved_dir = f"{saved_dir}/{path_src.name}.TranslatedBy."
        if self.translated_by:
            self.saved_dir = f"{saved_dir}{self.translated_by}"
            config = config + f"output: {self.saved_dir}\n"
        else:
            self.saved_dir = f"{saved_dir}{path_model.name}.outcome"
            config = config + f"output: {self.saved_dir}\n"
        config = config + f"min_length: {self.min_length}\n"
        config = config + f"verbose: {bool_yaml(self.verbose)}\n"
        if self.tgt:
            config = config + f"tgt: {self.tgt}\n"
        else:
            config = config + f"# tgt: {self.tgt}\n"
        if not self.enable_gpu:
            config = config + f"# gpu: 0\n"
        else:
            config = config + f"gpu: 0\n"

        config = config + f"replace_unk: {bool_yaml(self.replace_unk)}\n"
        config = config + f"# phrase_table: {self.phrase_table}\n"

        self.config = config


@dataclass
class TranslateExtra:
    """TranslateExtra is a data class for additional configuration options in sequence translation.

    Args:
        report_align (bool, optional): If True, report alignment for each translation. Defaults to False.
        gold_align (bool, optional): If True, report alignment between source and gold target,
            useful for testing the performance of learned alignments. Defaults to False.
        report_time (bool, optional): If True, report some translation time metrics. Defaults to False.
        attn_debug (bool, optional): If True, print the best attention for each word. Defaults to False.
        align_debug (bool, optional): If True, print the best alignment for each word. Defaults to False.
        with_score (bool, optional): If True, add a tab-separated score to the translation. Defaults to False.
        torch_profile (bool, optional): If True, report PyTorch profiling stats. Defaults to False.
        beam_size (int, optional): If beam size is 5, means the model considers the top 5 most likely next words at each decoding step.
                                   If beam size is 1, means the model only considers the most likely next word at each step.
                                        This is also known as greedy decoding
        dump_beam (str, optional): File to dump beam information to. Defaults to "".
        n_best (int, optional): If verbose is set, output the n_best decoded sentences. Defaults to 1.

    """

    report_align: bool = False
    gold_align: bool = False
    report_time: bool = False
    attn_debug: bool = False
    align_debug: bool = False
    with_score: bool = False
    torch_profile: bool = False
    beam_size: int = 5
    dump_beam: str = ""
    n_best: int = 1

    def __post_init__(self):
        config = f"\n\n## Inspect Translation\n\n"
        config = config + f"report_align: {bool_yaml(self.report_align)}\n"
        config = config + f"gold_align: {bool_yaml(self.gold_align)}\n"
        config = config + f"report_time: {bool_yaml(self.report_time)}\n"
        config = config + f"attn_debug: {bool_yaml(self.attn_debug)}\n"
        config = config + f"align_debug: {bool_yaml(self.align_debug)}\n"
        config = config + f"with_score: {bool_yaml(self.with_score)}\n"
        config = config + f"profile: {bool_yaml(self.torch_profile)}\n"
        config = config + f"beam_size: {self.beam_size}\n"
        if self.dump_beam:
            config = config + f"dump_beam: {self.dump_beam}\n"
        else:
            config = config + f"# dump_beam: {self.dump_beam}\n"
        config = config + f"n_best: {self.n_best}\n"
        self.config = config


def bool_yaml(val: bool) -> str:
    """
    Convert a Python boolean value into a YAML boolean representation.

    Args:
        val (bool): The boolean value to be converted.

    Returns:
        str: The YAML boolean representation of the input boolean.
             If the input is True, returns 'true'; if False, returns 'false'.

    Example:
        >>> bool_yaml(True)
        'true'
        >>> bool_yaml(False)
        'false'
    """
    return str(val).lower()


def compute_bleu(
    target_test: str,
    target_pred: str,
    n_gram: int = 2,
    lowercase: bool = True,
):
    """Compute BLEU score for machine translation.

    Args:
        target_test (str): The path to the detokenized file containing human translations.
        target_pred (str): The path to the detokenized file containing machine translations.
        n_gram (int, optional): The n-gram order for BLEU score computation. Defaults to 2.
        lowercase (bool, optional): Whether to convert reference and prediction texts to lowercase.
            Defaults to True.

    Returns:
        float: The computed BLEU score.

    Note:
        Make sure that the target_test and target_pred files are already detokenized.

    Example:
        bleu_score = compute_bleu("human_translations.txt", "machine_translations.txt")
    """
    # Make sure the target_test & target_pred files are already detokenized
    refs, preds = utils.load_eval_set(target_test, target_pred)

    # Initialize SacreBLEUScore metric
    sacre_bleu = SacreBLEUScore(n_gram=n_gram, lowercase=lowercase)

    # Compute BLEU score
    result = sacre_bleu(preds=preds, target=refs)

    # Convert PyTorch tensor to Python float
    result = result.item()

    return result


def compute_chrf(
    target_test: str,
    target_pred: str,
    word_order: int = 0,
    lowercase: bool = True,
) -> float:
    """Compute ChrF score for machine translation.

    Args:
        target_test (str): The path to the detokenized file containing human translations.
        target_pred (str): The path to the detokenized file containing machine translations.
        word_order (int, optional): The word order for ChrF score computation. Defaults to 0.
        lowercase (bool, optional): Whether to convert reference and prediction texts to lowercase.
            Defaults to True.

    Returns:
        float: The computed ChrF score.

    Note:
        Make sure that the target_test and target_pred files are already detokenized.

    Example:
        chrf_score = compute_chrf("human_translations.txt", "machine_translations.txt")
    """
    # Make sure the target_test & target_pred files are already detokenized
    refs, preds = utils.load_eval_set(target_test, target_pred)

    # Initialize CHRFScore metric
    chrf = CHRFScore(n_word_order=word_order, lowercase=lowercase)

    # Compute ChrF score
    result = chrf(preds=preds, target=refs)

    # Convert PyTorch tensor to Python float
    result = result.item()

    return result


def generate_config_translation(
    models_path: str,
    target_translation: str,
    translate_extra: TranslateExtra,
    translated_target: str = "",
    enable_gpu: bool = False,
) -> tuple:
    """
    Generate translation configuration files for a list of models.

    Args:
        models_path (str): The path to the directory containing model files.
        target_translation (str): The target language for translation.
        translate_extra (TranslateExtra): Receiving TranslateExtra dataclass, which is the translation debug configuration.
        translated_target (str): The original translated target sentence.
        enable_gpu (bool, optional): If True, enable GPU acceleration. Default is False.

    Returns:
        tuple: A tuple containing lists of configuration file paths, saved logs,
               directories for translated outputs, and model filenames.
    """
    models_l = os.listdir(models_path)
    # filter out non-model files
    models_l = [f"{models_path}/{file}" for file in models_l if file.endswith(".pt")]
    config_paths = []
    saved_logs = []
    save_translated = []

    saved_config_on = "./compilation/translate_config"
    utils.create_folder_if_not_exists(saved_config_on)
    for model in tqdm(models_l, desc="generate_config_translation()"):
        # generate translation config
        if translated_target:
            translate_essential = TranslateEssential(
                model=model,
                src=target_translation,
                tgt=translated_target,
                verbose=True,
                enable_gpu=enable_gpu,
            )
        else:
            translate_essential = TranslateEssential(
                model=model, src=target_translation, verbose=True, enable_gpu=enable_gpu
            )
        config_loc = f"{saved_config_on}/{Path(target_translation).name}_TRANSLATE_{Path(model).name}.yaml"
        config_paths.append(config_loc)
        saved_logs.append(f"{translate_essential.saved_dir}.log")
        save_translated.append(translate_essential.saved_dir)
        utils.write_to_file(
            config_loc, generateTrainingConfig(translate_essential, translate_extra)
        )

    # sort based on steps value
    config_paths = sorted(
        config_paths, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    saved_logs = sorted(saved_logs, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    save_translated = sorted(
        save_translated, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    models_l = sorted(models_l, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return config_paths, saved_logs, save_translated, models_l


def produce_translation(
    config_paths: List[str],
    saved_logs: List[str],
) -> None:
    """
    Execute translation for each specified configuration file.

    Args:
        config_paths (List[str]): List of paths to translation configuration files.
        saved_logs (List[str]): List of paths to saved log files.

    Example:
        produce_translation(
            config_paths=["/path/to/config1.yaml", "/path/to/config2.yaml"],
            saved_logs=["/path/to/log1.log", "/path/to/log2.log"],
        )
    """
    count_l = len(config_paths)
    for i in trange(count_l, desc="produce_translation()"):
        nmt_command = f"\
            onmt_translate -config {config_paths[i]} |& tee {saved_logs[i]}"
        command = ["bash", "-c", nmt_command]
        utils.execute_cmd(command)


def construct_desubwording(
    translated_target: str,
    subword_target_model: str,
    save_translated: List[str],
) -> Tuple[str, List[str]]:
    """
    Perform desubwording on translated sentences.

    Args:
        translated_target (str): The original translated target sentence.
        subword_target_model (str): The subword target model used for desubwording.
        save_translated (List[str]): List of translated sentences to perform desubwording on.

    Returns:
        Tuple[str, List[str]]: A tuple containing the desubworded original sentence
                              and a list of desubworded translated sentences.
    """
    translated_desubword = []
    real_translated = preprocess.sentence_desubword(
        target_model=subword_target_model, target_pred=translated_target
    )
    for translated in tqdm(save_translated, desc="construct_desubwording()"):
        translated_desubword.append(
            preprocess.sentence_desubword(
                target_model=subword_target_model, target_pred=translated
            )
        )
    return real_translated, translated_desubword


def make_evaluation(
    models_list: List[str],
    real_translated: str,
    translated_desubword: List[str],
) -> pd.DataFrame:
    """
    Perform evaluation on translated sentences using BLEU and CHR-F scores.

    Args:
        models_list (List[str]): List of model filenames used for translation.
        real_translated (str): The original translated target sentence (path).
        translated_desubword (List[str]): List of desubworded translated sentences (path).

    Returns:
        pd.DataFrame: A DataFrame containing evaluation scores (BLEU and CHR-F)
                      for each training steps.
    """
    # Compute evaluation scores
    df_score = {
        "steps": f_regex.extract_numbers_from_filenames(models_list),
        "bleu": [],
        "chrf": [],
    }

    for prediction in tqdm(translated_desubword, desc="make_evaluation()"):
        df_score["bleu"].append(
            compute_bleu(target_test=real_translated, target_pred=prediction)
        )
        df_score["chrf"].append(
            compute_chrf(target_test=real_translated, target_pred=prediction)
        )

    df_score = pd.DataFrame(df_score)
    df_score.sort_values(by="steps", inplace=True)
    df_score.reset_index(drop=True, inplace=True)
    return df_score


def perform_models_translation(
    models_path: str,
    tobe_translated: str,
    translated_target: str,
    subword_target_model: str,
    translate_extra: TranslateExtra,
    enable_gpu: bool = False,
) -> pd.DataFrame:
    """
    Perform translation, desubwording, and evaluation for a set of models.

    Args:
        models_path (str): The path to the directory containing model files.
        tobe_translated (str): The source sentence to be translated.
        translated_target (str): The original translated target sentence.
        subword_target_model (str): The subword target model used for desubwording.
        translate_extra (TranslateExtra): Receiving TranslateExtra dataclass, which is the translation debug configuration.
        enable_gpu (bool, optional): If True, enable GPU acceleration. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation scores (BLEU and CHR-F)
                      for each translated sentence.

    Example:
        df_evaluation = perform_models_translation(
            models_path="/path/to/models",
            tobe_translated="/path/to/source_sentences",
            translated_target="/path/to/target_sentences"",
            translate_extra="instance_of_TranslateExtra_dataclass",
            subword_target_model="/path/to/subword_model",
        )
    """
    # Generate translation configurations
    config_paths, saved_logs, save_translated, models_l = generate_config_translation(
        models_path=models_path,
        target_translation=tobe_translated,
        translate_extra=translate_extra,
        translated_target=translated_target,
        enable_gpu=enable_gpu,
    )

    # Perform NMT translation
    produce_translation(config_paths=config_paths, saved_logs=saved_logs)

    # Perform desubwording
    real_translated, translated_desubword = construct_desubwording(
        translated_target=translated_target,
        subword_target_model=subword_target_model,
        save_translated=save_translated,
    )

    # Compute evaluation scores
    df_score = make_evaluation(
        models_list=models_l,
        real_translated=real_translated,
        translated_desubword=translated_desubword,
    )

    return df_score


def sentence_level_evaluation(target_test: str, target_pred: str):
    """
    Evaluate sentence-level metrics for machine translation predictions.

    Args:
        target_test (str): Path to the file containing reference translations.
        target_pred (str): Path to the file containing predicted translations.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results with columns:
            - 'reference': List of reference translations.
            - 'prediction': List of predicted translations.
            - 'bleu': List of BLEU scores for each sentence pair.
            - 'chrf': List of ChrF scores for each sentence pair.
    """

    # Load reference and predicted translations
    refs, preds = utils.load_eval_set(target_test=target_test, target_pred=target_pred)

    # Initialize SacreBLEUScore metric
    sacre_bleu = SacreBLEUScore(n_gram=2, lowercase=True)

    # Initialize CHRFScore metric
    chrf = CHRFScore(n_word_order=0, lowercase=True)

    # Initialize DataFrame to store evaluation results
    df = {"reference": [], "prediction": [], "bleu": [], "chrf": []}

    # Compute metrics for each sentence pair
    for i in range(len(refs)):
        ref = [refs[i]]
        pred = [preds[i]]

        bleu_score = sacre_bleu(preds=pred, target=ref).item()
        chrf_score = chrf(preds=pred, target=ref).item()

        df["reference"].append(refs[i][0])
        df["prediction"].append(preds[i])
        df["bleu"].append(bleu_score)
        df["chrf"].append(chrf_score)

    # Convert the results to a DataFrame
    df = pd.DataFrame(df)
    return df


def gather_sentence_evaluation(target_pred_dir: str, target_test: str):
    """
    Gather sentence-level evaluation metrics for multiple translation files.

    Args:
        target_pred_dir (str): The directory containing multiple translation files from different model steps.
        target_test (str): Path to the file containing reference translations.

    Returns:
        Tuple: A tuple containing two elements:
            - list: A list of pandas DataFrames, each containing the evaluation results for a translation file.
            - list: A list of full paths to the translation files used in the evaluation.
    """

    file_list = os.listdir(target_pred_dir)

    # Filter files based on keywords from the target_test filename
    target_test_filename = Path(target_test).name
    target_test_filename = target_test_filename.split(".")
    file_list = utils.filter_files_by_keywords(
        file_list=file_list,
        keyword1=target_test_filename[-1],
        keyword2=target_test_filename[-2],
    )

    # sort in ascending order
    file_list = sorted(file_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Create full paths for the filtered files
    file_list = [f"{target_pred_dir}/{file}" for file in file_list]

    l_df = []

    for file in tqdm(file_list, desc="sentence level evaluation"):
        df = sentence_level_evaluation(target_test=target_test, target_pred=file)
        l_df.append(df)

    return l_df, file_list
