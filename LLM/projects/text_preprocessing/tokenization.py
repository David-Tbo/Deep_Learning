"""
Tokenization utilities using spaCy.

Tokenization is the process of splitting a text into elementary units
called tokens (words, punctuation marks, numbers, etc.).

This module uses spaCy because it provides language-aware tokenization
rules depending on the selected language model.
"""

import spacy


def load_spacy_model(model_name: str = "fr_core_news_sm"):
    """
    Load a spaCy language model.

    Parameters
    ----------
    model_name : str
        Name of the spaCy model.

    Returns
    -------
    spacy.Language
        Loaded spaCy pipeline.
    """
    return spacy.load(model_name)


def tokenize_text(
    text: str,
    nlp
) -> list[str]:
    """
    Tokenize a text using spaCy.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.Language
        Loaded spaCy language model.

    Returns
    -------
    list[str]
        List of tokens.
    """

    doc = nlp(text)

    return [token.text for token in doc]


def tokenize_with_attributes(
    text: str,
    nlp
) -> list[dict]:
    """
    Tokenize text and return token attributes.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.Language
        Loaded spaCy language model.

    Returns
    -------
    list[dict]
        Token information.
    """

    doc = nlp(text)

    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "is_stop": token.is_stop,
            "is_alpha": token.is_alpha
        }
        for token in doc
    ]


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    text = (
        "Les banques françaises utilisent des modèles "
        "de scoring pour prédire le risque de crédit."
    )

    # Load French spaCy model
    nlp = load_spacy_model("fr_core_news_sm")

    # Basic tokenization
    tokens = tokenize_text(text, nlp)

    print("Tokens:")
    print(tokens)

    # Tokenization with linguistic attributes
    print("\nToken attributes:")

    token_info = tokenize_with_attributes(text, nlp)

    for token in token_info:
        print(token)