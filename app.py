"""Streamlit app: compare topic modelling techniques in the browser.

    streamlit run app.py
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from topic_modelling import TECHNIQUES, Corpus, coherence, load_documents, run_technique, topic_diversity
from topic_modelling.techniques import TopicResult
from topic_modelling.visualization import comparison_chart, topic_bars, wordclouds

SAMPLE = Path(__file__).parent / "data" / "sample_corpus.txt"
CORE = [key for key, technique in TECHNIQUES.items() if not technique.optional]

st.set_page_config(page_title="Topic Modelling", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #2b1b63 0%, #6C5CE7 60%, #0984E3 100%);
    color: white; border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;
}
.hero h1 { color: white; margin: 0 0 6px 0; font-size: 2rem; }
.hero p { color: #efeaff; margin: 0; font-size: 1.05rem; }
.block-container { padding-top: 1.4rem; }
</style>
<div class="hero">
  <h1>🧠 Multiple Techniques for Topic Modelling</h1>
  <p>Paste text or load a dataset, then compare LDA, LSA, NMF and more side by side,
     with word clouds and coherence scores.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def sample_documents() -> list[str]:
    return load_documents(str(SAMPLE))


@st.cache_data(show_spinner="Downloading 20 Newsgroups…")
def newsgroup_documents(max_docs: int) -> list[str]:
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=42)
    return [doc for doc in data.data if len(doc.split()) > 20][:max_docs]


def read_upload(uploaded, text_column: str) -> list[str]:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    if uploaded.name.lower().endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        column = text_column if text_column in frame.columns else frame.columns[0]
        return [str(v).strip() for v in frame[column].dropna() if str(v).strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


@st.cache_data(show_spinner=False)
def analyse(documents: tuple[str, ...], keys: tuple[str, ...], num_topics: int,
            num_words: int, seed: int, stopwords: tuple[str, ...]):
    corpus = Corpus(list(documents), extra_stopwords=set(stopwords))
    _ = corpus.tokens  # fail fast on empty input before training
    rows, payloads = [], []
    for key in keys:
        result = run_technique(key, corpus, num_topics, num_words, seed)
        score = coherence(result, corpus)
        diversity = topic_diversity(result)
        rows.append({"Technique": result.name, "Key": key.upper(),
                     "Topics": len(result.topics),
                     "Coherence (NPMI)": None if math.isnan(score) else round(score, 3),
                     "Diversity": None if math.isnan(diversity) else round(diversity, 3),
                     "Time (s)": round(result.runtime, 2)})
        payloads.append({"key": key, "name": result.name, "note": result.note,
                         "topics": result.topics, "coherence": score, "diversity": diversity,
                         "runtime": result.runtime})
    return len(corpus), len(corpus.dictionary), rows, payloads


with st.sidebar:
    st.header("Settings")
    source = st.radio("Data", ["Sample corpus", "Paste text", "Upload file", "20 Newsgroups"])
    pasted, uploaded, column, max_docs = "", None, "", 500
    if source == "Paste text":
        pasted = st.text_area("One document per line", height=180,
                              placeholder="The match ended in a draw…\nInflation pushed prices higher…")
    elif source == "Upload file":
        uploaded = st.file_uploader("A .txt file (one document per line) or a .csv file")
        column = st.text_input("CSV text column", placeholder="leave blank to use the first column")
    elif source == "20 Newsgroups":
        max_docs = st.slider("Documents", 200, 2000, 500, 100)

    chosen = st.multiselect("Techniques", CORE, default=["lda", "lsa", "nmf"],
                            format_func=lambda key: TECHNIQUES[key].name)
    if TECHNIQUES["bertopic"].optional:
        st.caption("BERTopic is optional. Install it with `pip install bertopic` to enable it.")
    num_topics = st.slider("Topics", 2, 10, 4)
    num_words = st.slider("Words per topic", 5, 20, 10)
    extra_stops = st.text_input("Extra stopwords", placeholder="rt, amp, http")
    seed = st.number_input("Random seed", value=42, step=1)
    run = st.button("Discover topics", type="primary", width="stretch")

def show(fig):
    st.pyplot(fig)
    plt.close(fig)


if not run:
    st.info("Choose your data and techniques in the sidebar, then press **Discover topics**. "
            "The sample corpus has 40 short documents on AI research, sports, health and finance.")
    st.stop()

documents: list[str] = []
if source == "Sample corpus":
    documents = sample_documents()
elif source == "Paste text":
    documents = [line.strip() for line in pasted.splitlines() if line.strip()]
elif source == "Upload file" and uploaded is not None:
    documents = read_upload(uploaded, column.strip())
elif source == "20 Newsgroups":
    documents = newsgroup_documents(max_docs)

if len(documents) < 2:
    st.warning("Add at least two documents. In a text file or the paste box, put one document per line.")
    st.stop()
if not chosen:
    st.warning("Pick at least one technique.")
    st.stop()

stops = tuple(w.strip().lower() for w in extra_stops.replace(";", ",").split(",") if w.strip())
with st.spinner("Training models and drawing word clouds…"):
    try:
        n_docs, n_terms, rows, payloads = analyse(tuple(documents), tuple(chosen),
                                                  num_topics, num_words, int(seed), stops)
    except Exception as exc:
        st.error(f"Could not train the models: {exc}")
        st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", n_docs)
c2.metric("Unique terms", n_terms)
c3.metric("Topics", num_topics)
best = max(rows, key=lambda row: row["Coherence (NPMI)"] if row["Coherence (NPMI)"] is not None else -2)
c4.metric("Best coherence", best["Key"], help=best["Technique"])

st.subheader("Comparison")
frame = pd.DataFrame(rows)
st.dataframe(frame, width="stretch", hide_index=True)
show(comparison_chart([{"key": p["key"], "coherence": p["coherence"], "diversity": p["diversity"]}
                       for p in payloads]))

st.subheader("Topics")
for payload in payloads:
    result = TopicResult(payload["key"], payload["topics"], runtime=payload["runtime"],
                         note=payload["note"])
    score = "n/a" if payload["coherence"] != payload["coherence"] else f"{payload['coherence']:.3f}"
    with st.expander(f"{result.name}  ·  coherence {score}  ·  {result.runtime:.2f}s",
                     expanded=True):
        if result.note:
            st.caption(result.note)
        left, right = st.columns(2)
        with left:
            show(wordclouds(result))
        with right:
            show(topic_bars(result))
        st.write("  ·  ".join(f"**{i}.** " + ", ".join(words)
                              for i, words in enumerate(result.topic_words, 1)))
