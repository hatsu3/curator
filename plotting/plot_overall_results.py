from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "output" / "overall_results"


def load_results(json_path: str) -> pd.DataFrame:
    df = pd.read_json(json_path).T

    df["group"] = [x[0] for x in df.index.str.split("_", expand=True)]
    df["group"] = df["group"].replace(
        {
            "shared": "MF",
            "separate": "PT",
            "ours": "Curator",
        }
    )
    df["index_type"] = [x[1] for x in df.index.str.split("_", expand=True)]

    df["index_size_gb_avg"] = df["index_size_kb_avg"] / 1024 / 1024
    df["index_size_gb_std"] = df["index_size_kb_std"] / 1024 / 1024

    # use ms as the unit for insert and query latency
    for op_name in ["insert_grant", "query"]:
        for metric_type in ["std", "avg", "p50", "p99"]:
            df[f"{op_name}_lat_{metric_type}"] *= 1000

    # use µs as the unit for delete latency
    for op_name in ["delete"]:
        for metric_type in ["std", "avg", "p50", "p99"]:
            df[f"{op_name}_lat_{metric_type}"] *= 1000_000

    # convert qps to kqps
    for op_name in ["insert_grant", "query", "delete"]:
        for metric_type in ["avg", "std"]:
            df[f"{op_name}_qps_{metric_type}"] /= 1000

    return df


def plot_overall_results_single_metric(
    metric_name: str,
    arxiv_json_path: str = "plotting/data/overall_arxiv.json",
    yfcc_json_path: str = "plotting/data/overall_yfcc100m.json",
    fontsize: int = 14,
    figsize: tuple[int, int] = (7, 3.5),
    plot_p99: bool = False,
):
    """Plot the overall results for a single metric except for deletion latency.
    """
    
    arxiv_df = (
        load_results(arxiv_json_path)
        .reset_index()
        .rename(columns={"index": "index_name"})
    )
    yfcc_df = (
        load_results(yfcc_json_path)
        .reset_index()
        .rename(columns={"index": "index_name"})
    )
    arxiv_df["dataset"] = "arXiv"
    yfcc_df["dataset"] = "YFCC100M"

    combined_df = pd.concat([arxiv_df, yfcc_df], ignore_index=True)

    combined_df["index_type"] = combined_df["index_name"].map(
        lambda x: {
            "ours_ivf": "Curator",
            "shared_ivf": "IVF",
            "shared_hnsw": "HNSW",
            "separate_ivf": "IVF",
            "separate_hnsw": "HNSW",
        }[x]
    )

    combined_df["index_name"] = combined_df["index_name"].replace(
        {
            "ours_ivf": "Curator",
            "shared_ivf": "MF-IVF",
            "shared_hnsw": "MF-HNSW",
            "separate_ivf": "PT-IVF",
            "separate_hnsw": "PT-HNSW",
        }
    )

    print(combined_df)

    plt.rcParams.update({"font.size": fontsize})
    plt.figure(figsize=figsize)

    sns.barplot(
        data=combined_df,
        x="dataset",
        y=metric_name,
        hue="index_name",
        palette="tab10",
        order=["YFCC100M", "arXiv"],
        hue_order=["MF-IVF", "MF-HNSW", "PT-IVF", "PT-HNSW", "Curator"],
    )

    if plot_p99 and "lat" in metric_name:
        errorbar_metric_name = metric_name.replace("avg", "p99")
        # plot single-sided error bars
        for patch in plt.gca().patches:
            if not isinstance(patch, plt.Rectangle):
                continue

            row = combined_df[combined_df[metric_name] == patch.get_height()]
            if row.size == 0:
                continue

            p99 = row[errorbar_metric_name].values[0]
            upper_error = p99 - patch.get_height()
            plt.errorbar(
                x=patch.get_x() + patch.get_width() / 2,
                y=patch.get_height(),
                yerr=[[0], [upper_error]],
                color="gray",
            )

    plt.xlabel("")
    plt.ylabel(
        {
            "index_size_gb_avg": "Index size (GB)",
            "insert_grant_lat_avg": "Avg insert latency (ms)",
            "query_lat_avg": "Avg query latency (ms)",
            "delete_lat_avg": "Avg delete latency (µs)",
            "insert_grant_lat_p99": "P99 insert latency (ms)",
            "query_lat_p99": "P99 query latency (ms)",
            "delete_lat_p99": "P99 delete latency (µs)",
        }[metric_name]
    )

    plt.yscale("log")
    plt.grid(axis="y", which="major", linestyle="-", linewidth=0.5)

    legend = plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fontsize=fontsize - 2,
        ncol=5,
        columnspacing=1,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_DIR / f"overall_combined_{metric_name}.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


def plot_overall_results():
    plot_overall_results_single_metric("index_size_gb_avg")
    plot_overall_results_single_metric("query_lat_avg")
    plot_overall_results_single_metric("insert_grant_lat_avg")


def plot_deletion_results(
    json_path: str = "plotting/data/overall_deletion.json",
    fontsize: int = 14,
    figsize: tuple[int, int] = (7, 3),
    plot_p99: bool = False,
):
    combined_df = pd.read_json(json_path)
    combined_df["delete_lat_avg"] *= 1000_000
    combined_df["update_lat_avg"] *= 1000_000
    combined_df["delete_lat_p99"] *= 1000_000
    combined_df["update_lat_p99"] *= 1000_000

    combined_df["index_type"] = combined_df["index_type"].replace(
        {
            "curator": "Curator",
            "shared_ivf": "MF-IVF",
            "shared_hnsw": "MF-HNSW",
            "separate_ivf": "PT-IVF",
            "separate_hnsw": "PT-HNSW",
        }
    )

    combined_df["dataset_name"] = combined_df["dataset_name"].replace(
        {
            "arxiv_large": "arXiv",
            "yfcc100m": "YFCC100M",
        }
    )

    delete_df = combined_df[
        combined_df["index_type"].isin({"MF-IVF", "PT-IVF", "Curator"})
    ]
    update_df = combined_df[
        combined_df["index_type"].isin({"MF-HNSW", "PT-HNSW", "Curator"})
    ]

    color_map = {
        "MF-IVF": "tab:blue",
        "MF-HNSW": "tab:orange",
        "PT-IVF": "tab:green",
        "PT-HNSW": "tab:red",
        "Curator": "tab:purple",
    }

    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    indexes = ["MF-IVF", "PT-IVF", "Curator"]
    palette = [color_map[index] for index in indexes]
    sns.barplot(
        ax=axes[0],
        data=delete_df,
        x="dataset_name",
        y="delete_lat_avg",
        hue="index_type",
        palette=palette,
        order=["YFCC100M", "arXiv"],
        hue_order=indexes,
        legend=False,  # type: ignore
    )

    if plot_p99:
        errorbar_metric_name = "delete_lat_p99"
        for patch in axes[0].patches:
            if not isinstance(patch, plt.Rectangle):
                continue

            row = delete_df[delete_df["delete_lat_avg"] == patch.get_height()]
            if row.size == 0:
                continue

            p99 = row[errorbar_metric_name].values[0]
            upper_error = p99 - patch.get_height()
            axes[0].errorbar(
                x=patch.get_x() + patch.get_width() / 2,
                y=patch.get_height(),
                yerr=[[0], [upper_error]],
                color="gray",
            )

    if not plot_p99:
        axes[0].set_ylim(1, 40)

    axes[0].set_xlabel("")
    axes[0].set_ylabel("Avg deletion latency (µs)")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", which="major", linestyle="-", linewidth=0.5)

    indexes = ["MF-HNSW", "PT-HNSW", "Curator"]
    palette = [color_map[index] for index in indexes]
    sns.barplot(
        ax=axes[1],
        data=update_df,
        x="dataset_name",
        y="update_lat_avg",
        hue="index_type",
        palette=palette,
        order=["YFCC100M", "arXiv"],
        hue_order=indexes,
        legend=False,  # type: ignore
    )

    if plot_p99:
        errorbar_metric_name = "update_lat_p99"
        for patch in axes[1].patches:
            if not isinstance(patch, plt.Rectangle):
                continue

            row = update_df[update_df["update_lat_avg"] == patch.get_height()]
            if row.size == 0:
                continue

            p99 = row[errorbar_metric_name].values[0]
            upper_error = p99 - patch.get_height()
            axes[1].errorbar(
                x=patch.get_x() + patch.get_width() / 2,
                y=patch.get_height(),
                yerr=[[0], [upper_error]],
                color="gray",
            )

    axes[1].set_xlabel("")
    axes[1].set_ylabel("Avg update latency (µs)")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", which="major", linestyle="-", linewidth=0.5)

    bars = [
        axes[0].patches[0],
        axes[0].patches[2],
        axes[1].patches[0],
        axes[1].patches[2],
        axes[0].patches[4],
    ]
    labels = [
        "MF-IVF",
        "PT-IVF",
        "MF-HNSW",
        "PT-HNSW",
        "Curator",
    ]

    legend = fig.legend(
        bars,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fontsize=fontsize - 2,
        ncol=5,
        columnspacing=1,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_DIR / "overall_combined_delete_update.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    fire.Fire()
