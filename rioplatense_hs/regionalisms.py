import pandas as pd
from rapidfuzz import fuzz
from .preprocessing import remove_accents
import spacy

df_slang = pd.read_json("../data/dict_argentino.json").T

df_slang.drop(columns="term")

# Suffixes to convert
suffixes = {
    "ito": "o",
    "ita": "a",
    "itos": "os",
    "itas": "as",
}


def chop_suffixes(word):
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def post_process_tokens(toks):
    ret = []
    for tok in toks:
        tok = remove_accents(tok.lower())
        tok = chop_suffixes(tok)
        ret.append(tok)
    return ret


nlp = spacy.load("es_dep_news_trf")

threshold = 90


def find_slangs(text, slangs=None, fuzz_threshold=90):
    """
    Return the slangs used in the text.

    Parameters
    ----------

    text: str
        The text to analyze

    slangs: list
        List of slangs to search for. If None, slangs from the dictionary are used.

    fuzz_threshold: int
        The minimum similarity between the text and the slang to consider it used.
        Must be between 0 and 100.
    """

    if slangs is None:
        slangs = df_slang.index
    used_slangs = []

    doc = nlp(text)
    toks = [token.lemma_ for token in doc]
    toks = post_process_tokens(toks)

    # TODO: This is not fine -- but it works
    # We should use tok.whitespace_ to get the original token
    normalized_text = " ".join(toks)

    for slang in slangs:
        if fuzz.partial_ratio(f" {slang} ", normalized_text) >= fuzz_threshold:
            used_slangs.append(slang)

    return used_slangs
