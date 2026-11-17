"""Matplotlib charts and word clouds for topic words and model comparison."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .techniques import TopicResult  # noqa: E402

PALETTE = ["#6C5CE7", "#00B894", "#E17055", "#0984E3", "#FDCB6E",
           "#E84393", "#00CEC9", "#636E72"]
# Hue of each palette colour, so word-cloud text stays dark enough to read.
HUES = [262, 162, 16, 204, 42, 330, 178, 200]


def _topic_color(hue: int):
    def color(word, font_size, position, orientation, random_state=None, **kwargs):
        spread = (random_state.randint(0, 16) if random_state is not None else 8)
        return f"hsl({hue}, 68%, {30 + spread}%)"

    return color


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def topic_bars(result: TopicResult) -> plt.Figure:
    """One horizontal bar chart per topic showing its top words."""
    n = max(1, len(result.topics))
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i >= len(result.topics):
            ax.axis("off")
            continue
        words, weights = zip(*result.topics[i][::-1])
        ax.barh(words, [abs(w) for w in weights], color=PALETTE[i % len(PALETTE)])
        ax.set_title(f"Topic {i + 1}", fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(result.name, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def wordclouds(result: TopicResult) -> plt.Figure:
    """One word cloud per topic. Word size follows the model's word weight."""
    from wordcloud import WordCloud

    n = max(1, len(result.topics))
    cols = min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.5 * rows), squeeze=False)
    fig.patch.set_facecolor("white")

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(result.topics):
            continue
        freqs = {word: abs(weight) for word, weight in result.topics[i] if weight}
        ax.set_title(f"Topic {i + 1}", fontweight="bold", color=PALETTE[i % len(PALETTE)])
        if not freqs:
            continue
        cloud = WordCloud(width=900, height=480, background_color="white",
                          color_func=_topic_color(HUES[i % len(HUES)]),
                          prefer_horizontal=0.85, relative_scaling=0.4,
                          min_font_size=16, random_state=42).generate_from_frequencies(freqs)
        ax.imshow(cloud, interpolation="bilinear")

    fig.suptitle(f"{result.name}  ·  word clouds", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def comparison_chart(rows: list[dict]) -> plt.Figure:
    """Side-by-side bars of coherence and diversity for every technique."""
    names = [r["key"].upper() for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title, ylim in ((axes[0], "coherence", "Coherence (NPMI)", (-1, 1)),
                                    (axes[1], "diversity", "Topic diversity", (0, 1.1))):
        values = [0 if math.isnan(r[metric]) else r[metric] for r in rows]
        bars = ax.bar(names, values, color=PALETTE[: len(names)])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.axhline(0, color="#999", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Technique comparison (higher is better)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_topics(result: TopicResult, out_dir: str | Path) -> Path:
    return _save(topic_bars(result), Path(out_dir) / f"topics_{result.key}.png")


def plot_wordclouds(result: TopicResult, out_dir: str | Path) -> Path:
    return _save(wordclouds(result), Path(out_dir) / f"wordcloud_{result.key}.png")


def plot_comparison(rows: list[dict], out_dir: str | Path) -> Path:
    return _save(comparison_chart(rows), Path(out_dir) / "comparison.png")
