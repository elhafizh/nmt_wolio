from dataclasses import InitVar, dataclass, is_dataclass
from typing import List, Type, Union

from torchmetrics.text import CHRFScore, SacreBLEUScore, TranslationEditRate

from helpers import utils


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


def training(config_file: str) -> None:
    """Train a model based on the specified OpenNMT configuration file.

    Args:
        config_file (str): The file path to the OpenNMT configuration file.

    Note:
        This function uses the 'onmt_train' command to initiate the training
        process for a neural machine translation model. The training is
        configured using the provided OpenNMT configuration file.

    Example:
        >>> training("my_training_config.yaml")
        # Initiates model training based on the configuration specified in
        # 'my_training_config.yaml'.

    """
    command = ["onmt_train", "-config", config_file]
    utils.execute_cmd(command)


@dataclass
class TranslateEssential:
    """TranslateEssential is a data class for configuring sequence translation.

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
    """

    model: Union[str, List[str]]
    src: str
    translated_by: str = ""
    min_length: int = 1
    verbose: bool = False
    tgt: str = ""
    replace_unk: bool = False
    phrase_table: str = ""

    def __post_init__(self):
        config = f"\n## Data \n \n"
        if isinstance(self.model, list):
            # if isinstance(self.src, list) and all(isinstance(item, str) for item in self.model):
            config = config + f"model:\n"
            for i in range(len(self.model)):
                config = config + f"- {self.model[i]}\n"
        elif isinstance(self.model, str):
            config = config + f"model: {self.model}\n"
        config = config + f"src: {self.src}\n"
        if self.translated_by:
            config = config + f"output: {self.src}.translated.{self.translated_by}\n"
        else:
            config = config + f"output: {self.src}.translated\n"
        config = config + f"min_length: {self.min_length}\n"
        config = config + f"verbose: {bool_yaml(self.verbose)}\n"
        config = config + f"# tgt: {self.tgt}\n"
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

    return result
