from pysentimiento.preprocessing import preprocess_tweet as _pysent_preprocess
import re

labels = [
    "WOMEN",
    "LGBTI",
    "RACISM",
    "CLASS",
    "POLITICS",
    "DISABLED",
    "APPEARANCE",
    "CRIMINAL",
]

accent_replacements = {
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "Á": "A",
    "É": "E",
    "Í": "I",
    "Ó": "O",
    "Ú": "U",
}


def remove_accents(text):
    text = [accent_replacements.get(c, c) for c in text]
    text = "".join(text)

    return text


translations = {
    "WOMEN": "mujer",
    "LGBTI": "lgbti",
    "RACISM": "racismo",
    "CLASS": "clase",
    "POLITICS": "política",
    "DISABLED": "discapacidad",
    "APPEARANCE": "aspecto",
    "CRIMINAL": "criminal",
}


inv_translations = {remove_accents(v): k for k, v in translations.items()}


def label_to_text(row):
    """
    Converts a row with labels to a string with the labels separated by commas.

    Arguments:
        row: A dictionary with labels as keys and 1 or 0 as values.

    Returns:
        A string with the labels separated by commas.
    """
    ret = ""
    for label in labels:
        if row[label] == 1:
            new_label = translations[label]
            if ret == "":
                ret = new_label
            else:
                ret = ret + ", " + new_label
    if ret == "":
        ret = "nada"
    return ret


def text_to_label(text):
    """
    Converts a string with labels separated by commas to a row with labels.

    Arguments:
        text: A string with labels separated by commas.

    Returns:
        A dictionary with labels as keys and 1 or 0 as values.
    """

    # If "la respuesta final" in text => check after that
    # Else, check everywhere
    lower_text = text.lower()

    if "la respuesta final" in lower_text:
        index = lower_text.index("la respuesta final")
        lower_text = lower_text[index:]
        # Split at next .

        lower_text = lower_text.split(".")[0]

    lower_text = remove_accents(lower_text)

    row = {l: 0 for l in labels}

    for k, v in inv_translations.items():
        if k in lower_text:
            row[v] = 1
    return row


url_regex = r"\burl\b"


def preprocess_tweet(text):
    text = _pysent_preprocess(text, preprocess_hashtags=False, demoji=False)
    text = text.replace("@usuario", "")

    text = re.sub(url_regex, "", text)
    # Replace multiple spaces with one
    text = " ".join(text.split())
    text = text.replace("\n", " ")

    return text
