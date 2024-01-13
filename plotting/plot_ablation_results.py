from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "output" / "ablation"


def load_results(json_path: str) -> pd.DataFrame:
    df = pd.read_json(json_path)

    df["index_type"] = df["index_type"].replace(
        {
            "shared_ivf": "MF-IVF",
            "shared_ivf_bf": "+BF",
            "curator_flat": "+SL",
            "curator": "+BFS",
        }
    )

    df["dataset_name"] = df["dataset_name"].replace(
        {
            "yfcc100m": "YFCC100M",
            "arxiv_large": "arXiv",
        }
    )

    df["query_lat_avg"] = df["query_lat_avg"] * 1000
    df["index_size_gb_avg"] = df["index_size_kb_avg"] / 1024 / 1024

    return df


def plot_ablation_results(
    json_path: str = "plotting/data/ablation_component.json",
    fontsize: int = 14,
    figsize: tuple[int, int] = (7, 3),
):
    df = load_results(json_path)

    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.barplot(
        ax=axes[0],
        data=df[df["dataset_name"] == "YFCC100M"],
        x="index_type",
        y="query_lat_avg",
        order=["MF-IVF", "+BF", "+SL", "+BFS"],
    )

    axes[0].set_xlabel("")
    axes[0].set_ylabel("Avg query latency (ms)")
    axes[0].set_yscale("log")
    axes[0].set_title("YFCC100M")
    axes[0].grid(axis="y", which="major", linestyle="-", linewidth=0.5)

    sns.barplot(
        ax=axes[1],
        data=df[df["dataset_name"] == "arXiv"],
        x="index_type",
        y="query_lat_avg",
        order=["MF-IVF", "+BF", "+SL", "+BFS"],
    )

    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_yscale("log")
    axes[1].set_title("arXiv")
    axes[1].grid(axis="y", which="major", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"ablation_results.pdf")


if __name__ == "__main__":
    fire.Fire()
