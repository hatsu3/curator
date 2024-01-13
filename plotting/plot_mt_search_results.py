import json
from pathlib import Path

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "output" / "mt_search_results"


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
            "ours_ivf": "Curator",
        }
    )

    df["query_qps_avg"] = df["query_qps_avg"] / 1000  # convert to kQPS
    df["query_lat_avg"] = df["query_lat_avg"] * 1000  # convert to ms

    return df


def plot_mt_search_results(
    inter_json_path="plotting/data/mt_search_yfcc100m_inter.json",
    intra_json_path="plotting/data/mt_search_yfcc100m_intra.json",
    fontsize=14,
    figsize=(7, 3),
    linewidth=2,
):
    inter_df = load_results(inter_json_path)
    intra_df = load_results(intra_json_path)

    hnsw_mask = intra_df["index_type"].isin(["PT-HNSW", "MF-HNSW"])
    intra_df.loc[hnsw_mask, "query_lat_avg"] *= intra_df.loc[hnsw_mask, "num_threads"]

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

    for index_type, group in intra_df.groupby("index_type"):
        style = "--" if str(index_type).startswith("PT") else "-"
        color = color_map[str(index_type)]
        marker = marker_map[str(index_type).rsplit("-", maxsplit=1)[-1]]

        if index_type in ["MF-HNSW", "PT-HNSW"]:
            sns.lineplot(
                ax=axes[0],
                x=[1, 2, 4, 8, 16],
                y=group[group["num_threads"] == 1]["query_lat_avg"].values[0],
                marker=marker,
                label=index_type,
                linestyle=style,
                linewidth=linewidth,
                color=color,
                legend=False,  # type: ignore
            )
        else:
            sns.lineplot(
                ax=axes[0],
                data=group,
                x="num_threads",
                y="query_lat_avg",
                marker=marker,
                label=index_type,
                linestyle=style,
                linewidth=linewidth,
                color=color,
                legend=False,  # type: ignore
            )

    axes[0].grid(True, which="major", axis="y")
    axes[0].set_xlabel("Number of threads")
    axes[0].set_ylabel("Query latency (ms)")
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 2, 4, 8, 16])
    axes[0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[0].get_xaxis().set_tick_params(which="minor", size=0)
    axes[0].get_xaxis().set_tick_params(which="minor", width=0)
    axes[0].set_yscale("log")
    axes[0].set_title("Intra-query parallelism", y=1.05)

    for index_type, group in inter_df.groupby("index_type"):
        style = "--" if str(index_type).startswith("PT") else "-"
        color = color_map[str(index_type)]
        marker = marker_map[str(index_type).rsplit("-", maxsplit=1)[-1]]
        sns.lineplot(
            ax=axes[1],
            data=group,
            x="num_threads",
            y="query_qps_avg",
            marker=marker,
            label=index_type,
            linestyle=style,
            linewidth=linewidth,
            color=color,
            legend=False,  # type: ignore
        )

    axes[1].grid(True, which="major", axis="y")
    axes[1].set_xlabel("Number of threads")
    axes[1].set_ylabel(r"Avg query QPS ($\times 10^3$)")
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 2, 4, 8, 16])
    axes[1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[1].get_xaxis().set_tick_params(which="minor", size=0)
    axes[1].get_xaxis().set_tick_params(which="minor", width=0)
    axes[1].set_yscale("log")
    axes[1].set_title("Inter-query parallelism", y=1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = ["MF-IVF", "MF-HNSW", "PT-IVF", "PT-HNSW", "Curator"]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]

    legend = fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fontsize=fontsize - 2,
        ncol=5,
        columnspacing=1,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_DIR / "mt_search_results.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    fire.Fire()
