"""Metrics to compare topic models that were trained in different libraries."""
from __future__ import annotations

import math
import warnings

from .preprocessing import Corpus
from .techniques import TopicResult


def coherence(result: TopicResult, corpus: Corpus, measure: str = "c_npmi") -> float:
    """Topic coherence in [-1, 1] (higher is better); NaN when it cannot be computed.

    NPMI is the default because C_v rewards random word sets on small corpora.
    """
    from gensim.models import CoherenceModel

    vocab = corpus.dictionary.token2id
    topics = [[w for w in words if w in vocab] for words in result.topic_words]
    topics = [t for t in topics if len(t) >= 2]
    if not topics:
        return math.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CoherenceModel(topics=topics, texts=corpus.tokens,
                                   dictionary=corpus.dictionary, coherence=measure,
                                   processes=1)
            score = model.get_coherence()
    except Exception:
        return math.nan
    return float(score)


def topic_diversity(result: TopicResult) -> float:
    """Share of unique words across all topics' top words (1.0 = no overlap)."""
    words = [w for topic in result.topic_words for w in topic]
    return len(set(words)) / len(words) if words else math.nan
