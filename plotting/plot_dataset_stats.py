from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from benchmark.utils import get_dataset_config
from dataset import get_dataset, get_metadata

OUTPUT_DIR = Path(__file__).parent / "output" / "dataset_stats"


def preprocess_data(dataset: str = "arxiv-large-10", ndim=2, seed=42):
    assert dataset in ["arxiv-large-10", "yfcc100m"]

    dataset_config, dim = get_dataset_config(dataset, test_size=0)
    train_vecs, test_vecs, metadata = get_dataset(
        dataset_name=dataset_config.dataset_name, **dataset_config.dataset_params
    )
    train_mds, test_mds = get_metadata(
        synthesized=dataset_config.synthesize_metadata,
        train_vecs=train_vecs,
        test_vecs=test_vecs,
        dataset_name=dataset_config.dataset_name,
        **dataset_config.metadata_params,
    )

    scaler = StandardScaler()
    train_vecs = scaler.fit_transform(train_vecs)
    test_vecs = scaler.transform(test_vecs)

    pca = PCA(n_components=ndim, random_state=seed)
    train_vecs = pca.fit_transform(train_vecs)
    test_vecs = pca.transform(test_vecs)

    return train_vecs, test_vecs, train_mds, test_mds


def plot_per_tenant_combined(dataset="arxiv-large-10", figsize=(7, 3), fontsize=14):
    train_vecs, __, train_mds, __ = preprocess_data(dataset=dataset, ndim=1)

    plt.rcParams.update({"font.size": fontsize})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    n_tenants = 100 if dataset == "arxiv-large-10" else 1000

    n_vecs = np.zeros(n_tenants)
    for md in train_mds:
        for tenant_id in md:
            n_vecs[tenant_id] += 1

    n_vecs_pct = np.array([n_vec / len(train_vecs) * 100 for n_vec in n_vecs])

    sns.histplot(
        x=n_vecs_pct,
        bins=20,
        legend=False,
        ax=ax1,
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("% of per-tenant vectors")
    ax1.set_ylabel("# of tenants")

    sharing_degrees = np.array([len(mds) for mds in train_mds])
    sns.histplot(
        x=sharing_degrees,
        bins=20,
        legend=False,
        log_scale=(True, False),
        ax=ax2,
    )

    ax2.set_xlabel("Sharing degree")
    ax2.set_ylabel("# of vectors")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"{dataset}_per_tenant_combined.pdf")


if __name__ == "__main__":
    fire.Fire()
