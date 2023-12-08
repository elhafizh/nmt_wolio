import re


def delete_unnecessary_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def delete_words_from_pb(sentence: str) -> str:
    """
    Deletes the portion of the input sentence starting from the occurrence of "(pb)"
    and extending until the end of the sentence, including any punctuation.

    Examples:
        >>> delete_words_from_pb("This is a sample (pb) sentence.   ")
        'This is a sample'
    """
    # Define the pattern to match "(pb)" and everything after it until the end of the sentence
    pattern = re.compile(r"\(pb\)\s*.*?(?=[.!?]|$)")

    # Use sub() to replace the matched pattern with an empty string
    result = re.sub(pattern, "", sentence)

    # Remove trailing whitespace and the dot at the end
    result = result.rstrip(".").rstrip()

    return result


def delete_istl_from_sentence(sentence: str) -> str:
    """
    Deletes the exact string "(istl)" from the input sentence.

    Examples:
        >>> delete_istl_from_sentence("Sentence example (istl) another example.")
        'Sentence example another example.'
    """
    # Use replace() to remove the exact string "(istl)"
    result = sentence.replace("(istl)", "")
    result = delete_unnecessary_whitespace(result)

    return result.strip()


def remove_sentence_after_asterisk(input_text: str) -> str:
    """
    Removes the asterisk symbol (*) and the entire sentence that follows it from the input text.

    Args:
        input_text (str): The input text containing sentences and asterisk symbols.

    Returns:
        str: The cleaned text with the asterisk symbol and the following sentence removed.

    Example:
        >>> text = "There are many variations of passages *of Lorem Ipsum available"
        >>> cleaned_text = remove_sentence_after_asterisk(text)
        >>> print(cleaned_text)
        "There are many variations of passages"
    """
    cleaned_text = re.sub(r"\*.*", "", input_text)
    return cleaned_text


def is_hidden(name: str) -> bool:
    """Check if a file or folder is hidden based on its name.

    Args:
        name (str): The name of the file or folder.

    Returns:
        bool: True if the file or folder is hidden, False otherwise.
    """
    hidden_pattern = re.compile(r"^\.")

    return bool(hidden_pattern.match(name))
