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

translations = {
    "WOMEN": "mujer",
    "LGBTI": "lgbti",
    "RACISM": "racismo",
    "CLASS": "clase",
    "POLITICS": "política",
    "DISABLED": "discapacidad",
    "APPEARANCE": "apariencia",
    "CRIMINAL": "criminal",
}

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

    if "la respuesta final" in text.lower():
        index = text.lower().index("la respuesta final")
        text = text[index:]

    text = text.lower()

    row = {l: 0 for l in labels}
    for label in labels:
        if translations[label] in text:
            row[label] = 1
    return row
