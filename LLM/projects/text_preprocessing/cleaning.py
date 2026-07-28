"""
Text cleaning utilities.

This module provides functions to clean raw textual data before NLP tasks.
"""

import re


def remove_html(text: str) -> str:
    """
    Remove HTML tags from text.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Text without HTML tags.
    """
    return re.sub(r"<.*?>", " ", text)


def remove_extra_spaces(text: str) -> str:
    """
    Normalize multiple spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


def clean_text(
    text: str,
    remove_html_tags: bool = True
) -> str:
    """
    Apply basic text cleaning.

    Parameters
    ----------
    text : str
        Raw text.
    remove_html_tags : bool
        Remove HTML tags if True.

    Returns
    -------
    str
        Cleaned text.
    """

    if remove_html_tags:
        text = remove_html(text)

    text = remove_extra_spaces(text)

    return text


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    raw_text = """
    <html>
        Bonjour   le monde !
        Ceci est un exemple NLP.
    </html>
    """

    cleaned = clean_text(raw_text)

    print("Original text:")
    print(raw_text)

    print("\nCleaned text:")
    print(cleaned)