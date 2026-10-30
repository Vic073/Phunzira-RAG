"""Tokenisation, light normalisation and language identification for English/Chichewa text."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS_EN = frozenset(
    "a an the and or of to in on at by for with from is are was were be been it its "
    "this that these those as can do does how what why which who when where into than "
    "then so such not no".split()
)
STOPWORDS_NY = frozenset(
    "ndi ndipo komanso kuti ku mu pa wa la ya za cha kwa mwa ngati koma kapena amene "
    "zimene chifukwa kodi ndiye pamene ichi izi bwanji chiyani".split()
)
STOPWORDS = STOPWORDS_EN | STOPWORDS_NY

# Function words that are strong evidence for one language. The two sets do not overlap.
_EN_MARKERS = frozenset("the is are of and what how why in to that which with".split())
_NY_MARKERS = frozenset(
    "ndi ndipo kuti ku mu pa wa za kodi komanso zimene amene chifukwa ngati bwanji chiyani".split()
)

_IRREGULAR_PLURALS = {"leaves": "leaf"}


def normalize(text: str) -> str:
    """Unicode-normalise, lowercase and unify apostrophes."""
    text = unicodedata.normalize("NFKC", text).lower()
    return text.replace("’", "'")


def _stem(token: str) -> str:
    """Very light English plural stripping.

    Chichewa words rarely end in a consonant 's', so this is safe to apply to both languages.
    """
    if token in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[token]
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def analyze(text: str) -> list[str]:
    """Tokens used for lexical matching: stopwords removed, plurals stripped."""
    return [
        _stem(t)
        for t in tokenize(text)
        if t not in STOPWORDS and (len(t) > 1 or t.isdigit())
    ]


def detect_language(text: str) -> str:
    """Return "ny" (Chichewa) or "en" (English). Ties default to English."""
    tokens = tokenize(text)
    en = sum(t in _EN_MARKERS for t in tokens)
    ny = sum(t in _NY_MARKERS for t in tokens)
    return "ny" if ny > en else "en"
