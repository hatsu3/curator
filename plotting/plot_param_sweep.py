from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "output" / "param_sweep_results"


def load_results(json_path: str) -> pd.DataFrame:
    df = pd.read_json(json_path)

    df["index_type"] = df["index_type"].replace(
        {
            "shared_ivf": "MF-IVF",
            "shared_hnsw": "MF-HNSW",
            "separate_ivf": "PT-IVF",
            "separate_hnsw": "PT-HNSW",
            "curator": "Curator",
        }
    )

    df["query_lat_avg"] = df["query_lat_avg"] * 1000  # convert to ms

    return df


def plot_param_sweep_results_lat_vs_mem(
    json_path="plotting/data/param_sweep_results.json",
    fontsize=16,
    figsize=(7.5, 4),
    markersize=150,
):
    df = load_results(json_path)
    df = df[(df["recall_at_k"] > 0.93) & (df["recall_at_k"] < 0.96)]
    df["index_size_gb_avg"] = df["index_size_kb_avg"] / 1024 / 1024  # convert to GB
    df["mt_strategy"] = df["index_type"].replace(
        {
            "MF-IVF": "Metadata\nfiltering",
            "MF-HNSW": "Metadata\nfiltering",
            "PT-IVF": "Per-tenant\nindexing",
            "PT-HNSW": "Per-tenant\nindexing",
            "Curator": "Curator",
        }
    )

    print(
        df.groupby(["mt_strategy"])[["index_size_gb_avg", "query_lat_avg"]]
        .mean()
        .reset_index()
    )

    plt.rcParams.update({"font.size": fontsize})
    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df,
        x="index_size_gb_avg",
        y="query_lat_avg",
        style="mt_strategy",
        hue="mt_strategy",
        markers=True,
        palette="tab10",
        s=markersize,
    )

    plt.yscale("log")
    plt.ylim(0, 5)
    plt.xlabel("Memory consumption (GB)")
    plt.ylabel("Avg query latency (ms)")
    plt.grid(True, which="major")

    legend = plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.5, 1.03),
        fontsize=fontsize - 4,
        ncol=1,
        markerscale=1.3,
        labelspacing=0.8,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_DIR / "param_sweep_results_lat_vs_mem.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def plot_param_sweep_results_lat_vs_recall(
    json_path="plotting/data/param_sweep_results.json",
    fontsize=14,
    figsize=(7.5, 4),
    markersize=80,
):
    df = load_results(json_path)

    plt.rcParams.update({"font.size": fontsize})
    plt.figure(figsize=figsize)

    index_order = ["MF-IVF", "MF-HNSW", "PT-IVF", "PT-HNSW", "Curator"]

    sns.scatterplot(
        data=df,
        x="recall_at_k",
        y="query_lat_avg",
        style="index_type",
        hue="index_type",
        hue_order=index_order,
        style_order=index_order,
        markers=True,  # can specify a list of markers
        palette="tab10",
        s=markersize,
    )

    plt.xlabel("Recall@10")
    plt.ylabel("Query latency (ms)")

    plt.yscale("log")
    plt.grid(True, which="major")

    legend = plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.5, 1.03),
        fontsize=fontsize - 2,
        ncol=1,
        markerscale=1.5,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_DIR / "param_sweep_results_lat_vs_recall.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    fire.Fire()
