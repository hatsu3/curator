import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "output" / "scalability"


def load_results(json_path: str) -> pd.DataFrame:
    data = json.load(open(json_path))

    partial_dfs = []
    for key in data:
        df = pd.DataFrame(data[key])
        df["index_type"] = key
        partial_dfs.append(df)

    df = pd.concat(partial_dfs, ignore_index=True)

    df["index_type"] = df["index_type"].replace(
        {
            "shared_ivf": "MF-IVF",
            "shared_hnsw": "MF-HNSW",
            "separate_ivf": "PT-IVF",
            "separate_hnsw": "PT-HNSW",
            "shared_curator": "Curator",
        }
    )

    return df


def plot_scalability_results(
    nvecs_json_path="plotting/data/scalability_nvec_yfcc100m.json",
    ntenants_json_path="plotting/data/scalability_ntnt_yfcc100m.json",
    fontsize=14,
    figsize=(7, 3),
    linewidth=2,
):
    nvecs_df = load_results(nvecs_json_path)
    nvecs_df["num_vectors"] = nvecs_df["num_vectors"] / 1000  # convert to k
    nvecs_df["query_lat_avg"] = nvecs_df["query_lat_avg"] * 1000 * 1000  # convert to µs

    ntenants_df = load_results(ntenants_json_path)
    ntenants_df["index_size_gb_avg"] = (
        ntenants_df["index_size_kb_avg"] / 1000 / 1000  # convert to GB
    )

    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    color_map = {
        "MF-IVF": "tab:blue",
        "MF-HNSW": "tab:orange",
        "PT-IVF": "tab:green",
        "PT-HNSW": "tab:red",
        "Curator": "tab:purple",
    }

    marker_map = {
        "IVF": "o",
        "HNSW": "s",
        "Curator": "D",
    }

    for index_type, group in nvecs_df.groupby("index_type"):
        style = "--" if str(index_type).startswith("PT") else "-"
        color = color_map[str(index_type)]
        marker = marker_map[str(index_type).rsplit("-", maxsplit=1)[-1]]
        sns.lineplot(
            ax=axes[0],
            data=group,
            x="num_vectors",
            y="query_lat_avg",
            marker=marker,
            label=index_type,
            linestyle=style,
            linewidth=linewidth,
            color=color,
            legend=False,  # type: ignore
        )

    axes[0].grid(True, which="major", axis="y")
    axes[0].set_xlabel("Number of vectors (k)")
    axes[0].set_ylabel("Query latency (µs)")
    axes[0].set_yscale("log")

    for index_type, group in ntenants_df.groupby("index_type"):
        style = "--" if str(index_type).startswith("PT") else "-"
        color = color_map[str(index_type)]
        marker = marker_map[str(index_type).rsplit("-", maxsplit=1)[-1]]
        sns.lineplot(
            ax=axes[1],
            data=group,
            x="num_tenants",
            y="index_size_gb_avg",
            marker=marker,
            label=index_type,
            linestyle=style,
            linewidth=linewidth,
            color=color,
            legend=False,  # type: ignore
        )

    axes[1].grid(True, which="major", axis="y")
    axes[1].set_xlabel("Number of tenants")
    axes[1].set_ylabel("Index size (GB)")
    axes[1].set_yscale("log")
    axes[1].set_ylim([1, 20])

    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    reordered_labels = ["MF-IVF", "MF-HNSW", "PT-IVF", "PT-HNSW", "Curator"]
    reordered_handles = [label_to_handle[label] for label in reordered_labels]

    legend = fig.legend(
        reordered_handles,
        reordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fontsize=fontsize - 2,
        ncol=5,
        columnspacing=1,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_DIR / "scalability_combined.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    fire.Fire()
