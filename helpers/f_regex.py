import re


def delete_unnecessary_whitespace(text: str):
    return re.sub(r"\s+", " ", text).strip()


def delete_words_from_pb(sentence: str):
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


def delete_istl_from_sentence(sentence: str):
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
