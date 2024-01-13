import numpy as np
import psutil

from dataset.arxiv_dataset import ARXIV_DATA_DIR, ArxivDatasetConfig
from dataset.randperm_dataset import RandPermDatasetConfig
from dataset.yfcc100m_dataset import YFCC100M_DIR, YFCC100MDatasetConfig


def recall(results, ground_truth):
    recall = list()
    for res, gt in zip(results, ground_truth):
        res, gt = set(res), set(gt)
        if -1 in gt:
            gt.remove(-1)
        if len(res) == 0:
            recall.append(0 if len(gt) != 0 else 1)
        else:
            recall.append(len(res.intersection(gt)) / len(res))
    return np.mean(recall)


def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024


def get_dataset_config(dataset_key: str, test_size: float = 0.2):
    if dataset_key == "arxiv-small":
        dataset_config = ArxivDatasetConfig(
            dataset_params={
                "embed_path": ARXIV_DATA_DIR / "embeddings.npy",
                "test_size": test_size,
            },
            synthesize_metadata=False,
            metadata_params={
                "pkl_path": ARXIV_DATA_DIR / "processed.pkl",
                "share_degree": None,
                "test_size": test_size,
            },
        )
        dim = 384
    elif dataset_key.startswith("arxiv-large"):
        if dataset_key == "arxiv-large":
            share_degree = None
        else:
            share_degree = int(dataset_key.split("-")[-1])
        dataset_config = ArxivDatasetConfig(
            dataset_params={
                "embed_path": ARXIV_DATA_DIR / "embeddings_user100_vec2e6.npy",
                "test_size": test_size,
            },
            synthesize_metadata=False,
            metadata_params={
                "pkl_path": ARXIV_DATA_DIR / "processed_user100_vec2e6.pkl",
                "share_degree": share_degree,
                "test_size": test_size,
            },
        )
        dim = 384
    elif dataset_key == "yfcc100m":
        dataset_config = YFCC100MDatasetConfig(
            dataset_params={
                "data_path": YFCC100M_DIR
                / "yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy",
                "test_size": test_size,
            },
            synthesize_metadata=False,
            metadata_params={
                "data_path": YFCC100M_DIR
                / "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl",
                "test_size": test_size,
            },
        )
        dim = 192
    elif dataset_key.startswith("randperm"):
        from dataset import get_dataset  # pylint: disable=import-outside-toplevel

        # dataset_key: "randperm-v{n_vectors}-t{n_tenants}-d{access_matrix_density}-{real_ds_name}"
        n_vectors = int(dataset_key.split("-")[1][1:])
        n_tenants = int(dataset_key.split("-")[2][1:])
        access_matrix_density = float(dataset_key.split("-")[3][1:])
        real_ds_name = "-".join(dataset_key.split("-")[4:])

        real_ds_config, dim = get_dataset_config(real_ds_name, test_size=test_size)
        train_vecs, test_vecs, _ = get_dataset(
            dataset_name=real_ds_config.dataset_name,
            **real_ds_config.dataset_params,
        )

        n_vectors_total = train_vecs.shape[0] + test_vecs.shape[0]
        if n_vectors > n_vectors_total:
            raise ValueError(
                f"n_vectors={n_vectors} is larger than the total number of vectors "
                f"in the dataset: {n_vectors_total}"
            )

        np.random.seed(42)
        train_size = int(n_vectors * (1 - test_size))
        train_vecs = train_vecs[np.random.choice(train_vecs.shape[0], train_size)]
        test_vecs = test_vecs[np.random.choice(test_vecs.shape[0], n_vectors - train_size)]

        dataset_config = RandPermDatasetConfig(
            dataset_params={
                "real_ds_name": real_ds_config.dataset_name,
                "randperm_n_vectors": n_vectors,
                "randperm_test_size": test_size,
                **real_ds_config.dataset_params,
            },
            synthesize_metadata=False,
            metadata_params={
                "n_vecs": n_vectors,
                "n_tenants": n_tenants,
                "access_matrix_density": access_matrix_density,
                "test_size": test_size,
                "seed": 42,
            },
        )

    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    return dataset_config, dim
