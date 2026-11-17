"""Topic modelling techniques behind a single, common interface.

Every technique takes a :class:`Corpus` and returns a :class:`TopicResult`, so
models from gensim, scikit-learn and BERTopic can be compared side by side.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .preprocessing import Corpus

Topic = list[tuple[str, float]]


@dataclass
class TopicResult:
    key: str
    topics: list[Topic]
    doc_topics: np.ndarray | None = None
    runtime: float = 0.0
    note: str = ""

    @property
    def name(self) -> str:
        return TECHNIQUES[self.key].name

    @property
    def topic_words(self) -> list[list[str]]:
        return [[w for w, _ in topic] for topic in self.topics]


@dataclass(frozen=True)
class Technique:
    key: str
    name: str
    family: str
    summary: str
    run: Callable[[Corpus, int, int, int], TopicResult]
    optional: bool = False


def _vectorize(corpus: Corpus, kind: str = "tfidf"):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    cls = TfidfVectorizer if kind == "tfidf" else CountVectorizer
    vectorizer = cls(analyzer=lambda tokens: tokens)
    matrix = vectorizer.fit_transform(corpus.tokens)
    return matrix, vectorizer.get_feature_names_out()


def _top_terms(components: np.ndarray, terms: np.ndarray, num_words: int) -> list[Topic]:
    topics = []
    for row in components:
        idx = row.argsort()[::-1][:num_words]
        topics.append([(str(terms[i]), float(row[i])) for i in idx])
    return topics


def _clamp_topics(num_topics: int, matrix) -> int:
    return max(1, min(num_topics, matrix.shape[0], matrix.shape[1] - 1))


# --------------------------------------------------------------------------- #
# Probabilistic models
# --------------------------------------------------------------------------- #
def run_lda(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    from gensim.models import LdaModel

    model = LdaModel(corpus.bow, num_topics=num_topics, id2word=corpus.dictionary,
                     passes=50, iterations=200, alpha="auto", eta="auto",
                     random_state=seed)
    topics = [model.show_topic(i, topn=num_words) for i in range(num_topics)]
    doc_topics = np.array([[p for _, p in model.get_document_topics(b, minimum_probability=0)]
                           for b in corpus.bow])
    return TopicResult("lda", topics, doc_topics)


def run_plsa(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    # NMF minimising KL divergence on raw counts is equivalent to pLSA
    # (Gaussier & Goutte, 2005).
    from sklearn.decomposition import NMF

    counts, terms = _vectorize(corpus, "count")
    k = _clamp_topics(num_topics, counts)
    model = NMF(n_components=k, beta_loss="kullback-leibler", solver="mu",
                init="nndsvda", max_iter=1000, random_state=seed)
    doc_topics = model.fit_transform(counts)
    word_given_topic = model.components_ / model.components_.sum(axis=1, keepdims=True)
    return TopicResult("plsa", _top_terms(word_given_topic, terms, num_words), doc_topics)


def run_hdp(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    from gensim.models import HdpModel

    model = HdpModel(corpus.bow, id2word=corpus.dictionary, random_state=seed)
    weights: dict[int, float] = {}
    for bow in corpus.bow:
        for topic_id, prob in model[bow]:
            weights[topic_id] = weights.get(topic_id, 0.0) + prob
    active = sorted(weights, key=weights.get, reverse=True)
    shown = active[:num_topics]
    topics = [[(w, float(p)) for w, p in model.show_topic(t, topn=num_words)] for t in shown]
    note = f"HDP infers the topic count itself: {len(active)} active, top {len(shown)} shown"
    return TopicResult("hdp", topics, note=note)


# --------------------------------------------------------------------------- #
# Matrix factorisation models
# --------------------------------------------------------------------------- #
def run_lsa(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    from sklearn.decomposition import TruncatedSVD

    tfidf, terms = _vectorize(corpus)
    k = _clamp_topics(num_topics, tfidf)
    model = TruncatedSVD(n_components=k, n_iter=100, random_state=seed)
    doc_topics = model.fit_transform(tfidf)
    return TopicResult("lsa", _top_terms(model.components_, terms, num_words), doc_topics)


def run_nmf(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    from sklearn.decomposition import NMF

    tfidf, terms = _vectorize(corpus)
    k = _clamp_topics(num_topics, tfidf)
    model = NMF(n_components=k, init="nndsvd", max_iter=500, random_state=seed)
    doc_topics = model.fit_transform(tfidf)
    return TopicResult("nmf", _top_terms(model.components_, terms, num_words), doc_topics)


# --------------------------------------------------------------------------- #
# Clustering / embedding models
# --------------------------------------------------------------------------- #
def run_kmeans(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    from sklearn.cluster import KMeans

    tfidf, terms = _vectorize(corpus)
    k = max(1, min(num_topics, tfidf.shape[0]))
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(tfidf)
    return TopicResult("kmeans", _top_terms(model.cluster_centers_, terms, num_words),
                       np.eye(k)[labels])


def run_bertopic(corpus: Corpus, num_topics: int, num_words: int, seed: int) -> TopicResult:
    try:
        from bertopic import BERTopic
        from sklearn.cluster import KMeans
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("BERTopic is optional; install it with `pip install bertopic`") from exc

    n_docs = len(corpus)
    k = max(1, min(num_topics, n_docs - 1))
    umap_model = UMAP(n_neighbors=max(2, min(15, n_docs - 1)),
                      n_components=max(2, min(5, n_docs - 2)),
                      min_dist=0.0, metric="cosine", random_state=seed)
    model = BERTopic(umap_model=umap_model,
                     hdbscan_model=KMeans(n_clusters=k, n_init=10, random_state=seed),
                     top_n_words=num_words)
    # Pass cleaned text so c-TF-IDF keywords share the vocabulary of other models.
    model.fit([" ".join(t) for t in corpus.tokens])
    topic_ids = sorted(t for t in model.get_topics() if t != -1)
    topics = [[(w, float(s)) for w, s in model.get_topic(t)[:num_words]] for t in topic_ids]
    return TopicResult("bertopic", topics)


TECHNIQUES: dict[str, Technique] = {t.key: t for t in (
    Technique("lda", "Latent Dirichlet Allocation", "Probabilistic",
              "Generative model: documents are mixtures of topics, topics are "
              "distributions over words.", run_lda),
    Technique("lsa", "Latent Semantic Analysis", "Matrix factorisation",
              "Truncated SVD of the TF-IDF matrix captures latent concepts.", run_lsa),
    Technique("nmf", "Non-Negative Matrix Factorization", "Matrix factorisation",
              "Factorises TF-IDF into non-negative, additive parts; very "
              "interpretable topics.", run_nmf),
    Technique("plsa", "Probabilistic Latent Semantic Analysis", "Probabilistic",
              "Probabilistic version of LSA, fit as KL-divergence NMF on word counts.",
              run_plsa),
    Technique("hdp", "Hierarchical Dirichlet Process", "Probabilistic (non-parametric)",
              "Bayesian non-parametric LDA that infers the number of topics.", run_hdp),
    Technique("kmeans", "K-Means Clustering", "Clustering",
              "Hard-clusters TF-IDF vectors; centroid terms describe each topic.",
              run_kmeans),
    Technique("bertopic", "BERTopic", "Neural embeddings",
              "Sentence-transformer embeddings + UMAP + clustering + class-based TF-IDF.",
              run_bertopic, optional=True),
)}


def run_technique(key: str, corpus: Corpus, num_topics: int = 4,
                  num_words: int = 8, seed: int = 42) -> TopicResult:
    start = time.perf_counter()
    result = TECHNIQUES[key].run(corpus, num_topics, num_words, seed)
    result.runtime = time.perf_counter() - start
    return result
