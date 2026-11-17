"""
@author: Noman Raza Shah

Run and compare multiple topic modelling techniques from the command line.

    python Topic_Modelling_techniques.py                       # all techniques, sample data
    python Topic_Modelling_techniques.py -t lda nmf -k 5 --plot
    python Topic_Modelling_techniques.py --input my_docs.txt
    python Topic_Modelling_techniques.py --dataset 20newsgroups --max-docs 2000
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from topic_modelling import TECHNIQUES, Corpus, coherence, load_documents, run_technique, topic_diversity

SAMPLE_DATA = Path(__file__).parent / "data" / "sample_corpus.txt"
console = Console()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare topic modelling techniques.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-i", "--input", help=".txt (one document per line) or .csv file")
    source.add_argument("-d", "--dataset", choices=["sample", "20newsgroups"], default="sample",
                        help="built-in dataset to use when --input is not given")
    parser.add_argument("--text-column", help="CSV column holding the text")
    parser.add_argument("--max-docs", type=int, default=1000,
                        help="documents to sample from 20newsgroups")
    parser.add_argument("-t", "--techniques", nargs="+", default=["all"],
                        choices=["all", *TECHNIQUES], metavar="NAME",
                        help=f"any of: all, {', '.join(TECHNIQUES)}")
    parser.add_argument("-k", "--num-topics", type=int, default=4)
    parser.add_argument("-w", "--num-words", type=int, default=8)
    parser.add_argument("--stopwords", nargs="+", default=[], metavar="WORD",
                        help="extra domain-specific words to ignore")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="save charts to --output-dir")
    parser.add_argument("-o", "--output-dir", default="outputs")
    parser.add_argument("--list", action="store_true", help="list techniques and exit")
    return parser.parse_args(argv)


def get_documents(args) -> list[str]:
    if args.input:
        return load_documents(args.input, args.text_column)
    if args.dataset == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups

        data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                                  shuffle=True, random_state=42)
        return [d for d in data.data if len(d.split()) > 20][: args.max_docs]
    return load_documents(str(SAMPLE_DATA))


def show_techniques() -> None:
    table = Table(title="Available techniques", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Key", style="bold cyan")
    table.add_column("Technique")
    table.add_column("Family", style="green")
    table.add_column("How it works")
    for t in TECHNIQUES.values():
        table.add_row(t.key, t.name + (" [dim](optional)[/]" if t.optional else ""),
                      t.family, t.summary)
    console.print(table)


def show_result(result) -> None:
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Top words")
    for i, words in enumerate(result.topic_words, 1):
        table.add_row(str(i), "  ".join(words))
    if result.note:
        table.caption = f"[italic]{result.note}[/]"
    subtitle = f"{TECHNIQUES[result.key].family} · {result.runtime:.2f}s"
    console.print(Panel(table, title=f"[bold]{result.name}[/] ({result.key.upper()})",
                        subtitle=f"[dim]{subtitle}[/]", border_style="magenta"))


def show_comparison(rows: list[dict]) -> None:
    def fmt(x):
        return "n/a" if math.isnan(x) else f"{x:.3f}"

    best = max((r for r in rows if not math.isnan(r["coherence"])),
               key=lambda r: r["coherence"], default=None)
    table = Table(title="Comparison", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Technique", style="bold cyan", no_wrap=True)
    for col in ("Topics", "Coherence (NPMI)", "Diversity", "Time (s)"):
        table.add_column(col, justify="right")
    for r in rows:
        star = " [yellow]★[/]" if r is best else ""
        table.add_row(r["key"].upper() + star, str(r["topics"]), fmt(r["coherence"]),
                      fmt(r["diversity"]), f"{r['runtime']:.2f}")
    console.print(table)


def main(argv=None) -> int:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args(argv)
    if args.list:
        show_techniques()
        return 0

    keys = ([k for k, t in TECHNIQUES.items() if not t.optional]
            if "all" in args.techniques else list(dict.fromkeys(args.techniques)))

    with console.status("Loading and preprocessing documents..."):
        corpus = Corpus(get_documents(args), extra_stopwords={w.lower() for w in args.stopwords})
        _ = corpus.bow
    console.print(Panel.fit(
        f"[bold]{len(corpus)}[/] documents · [bold]{len(corpus.dictionary)}[/] unique terms · "
        f"[bold]{args.num_topics}[/] topics · techniques: [cyan]{', '.join(keys)}[/]",
        title="[bold magenta]Topic Modelling[/]"))

    if args.plot:
        from topic_modelling.visualization import plot_comparison, plot_topics, plot_wordclouds

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for key in keys:
        try:
            with console.status(f"Training {TECHNIQUES[key].name}..."):
                result = run_technique(key, corpus, args.num_topics, args.num_words, args.seed)
        except ImportError as exc:
            console.print(f"[yellow]Skipping {key}:[/] {exc}")
            continue
        show_result(result)
        rows.append({"key": key, "name": result.name, "topics": len(result.topics),
                     "coherence": coherence(result, corpus),
                     "diversity": topic_diversity(result), "runtime": result.runtime})
        if args.plot:
            plot_topics(result, args.output_dir)
            plot_wordclouds(result, args.output_dir)

    if rows:
        show_comparison(rows)
    if args.plot and rows:
        plot_comparison(rows, args.output_dir)
        console.print(f"[green]Charts and word clouds saved to[/] {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
