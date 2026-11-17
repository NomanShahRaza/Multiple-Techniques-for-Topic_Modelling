"""Text cleaning shared by every technique, so results are directly comparable."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property

from gensim import corpora

_TOKEN_RE = re.compile(r"[a-z]+")


def _load_nltk():
    """Return (stopwords, lemmatize_fn), downloading NLTK data on first use."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem.wordnet import WordNetLemmatizer

        for resource, path in (("stopwords", "corpora/stopwords"),
                               ("wordnet", "corpora/wordnet"),
                               ("omw-1.4", "corpora/omw-1.4")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("warmup")
        return set(stopwords.words("english")), lemmatizer.lemmatize
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(ENGLISH_STOP_WORDS), lambda w: w


@dataclass
class Corpus:
    """Raw documents plus the cleaned representations each library expects."""

    documents: list[str]
    min_word_length: int = 3
    min_df: int | None = None
    max_df: float = 0.5
    extra_stopwords: set[str] = field(default_factory=set)

    @cached_property
    def tokens(self) -> list[list[str]]:
        stop, lemmatize = _load_nltk()
        stop |= self.extra_stopwords
        cleaned = []
        for doc in self.documents:
            # Check stopwords before lemmatizing too, otherwise "does" becomes "doe".
            words = (lemmatize(w) for w in _TOKEN_RE.findall(doc.lower()) if w not in stop)
            cleaned.append([w for w in words
                            if len(w) >= self.min_word_length and w not in stop])

        # Document-frequency filtering needs enough documents to be meaningful;
        # dropping one-off words from a small corpus removes most of the vocabulary.
        if len(cleaned) >= 10:
            min_df = self.min_df or (2 if len(cleaned) >= 100 else 1)
            df = Counter(w for doc in cleaned for w in set(doc))
            limit = self.max_df * len(cleaned)
            keep = {w for w, n in df.items() if min_df <= n <= limit}
            cleaned = [[w for w in doc if w in keep] for doc in cleaned]
        return cleaned

    @cached_property
    def dictionary(self) -> corpora.Dictionary:
        return corpora.Dictionary(self.tokens)

    @cached_property
    def bow(self) -> list[list[tuple[int, int]]]:
        return [self.dictionary.doc2bow(doc) for doc in self.tokens]

    def __len__(self) -> int:
        return len(self.documents)


def load_documents(path: str, text_column: str | None = None) -> list[str]:
    """Load a ``.txt`` file (one document per line) or a ``.csv`` file."""
    if path.lower().endswith(".csv"):
        import csv

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            column = text_column or reader.fieldnames[0]
            docs = [row[column] for row in reader]
    else:
        with open(path, encoding="utf-8") as fh:
            docs = fh.read().splitlines()
    return [d.strip() for d in docs if d.strip()]
