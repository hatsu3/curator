from pathlib import Path

import numpy as np

from dataset.metrics import Metric

DATA_DIR = Path(__file__).parent.parent / "data"

Metadata = list[list[int]]


def get_dataset(
    dataset_name: str, **dataset_kwargs
) -> tuple[np.ndarray, np.ndarray, dict]:
    if dataset_name == "arxiv":
        from dataset.arxiv_dataset import load_arxiv_dataset_vecs

        train_vecs, test_vecs = load_arxiv_dataset_vecs(**dataset_kwargs)
        metadata = {
            "metric": Metric.EUCLIDEAN,
            "dimension": train_vecs.shape[1],
        }

    elif dataset_name == "yfcc100m":
        from dataset.yfcc100m_dataset import load_yfcc_dataset_vecs

        train_vecs, test_vecs = load_yfcc_dataset_vecs(**dataset_kwargs)
        metadata = {
            "metric": Metric.EUCLIDEAN,
            "dimension": train_vecs.shape[1],
        }

    elif dataset_name == "randperm":
        real_ds_name = dataset_kwargs.pop("real_ds_name")
        n_vectors = dataset_kwargs.pop("randperm_n_vectors")
        test_size = dataset_kwargs.pop("randperm_test_size")
        train_vecs, test_vecs, metadata = get_dataset(real_ds_name, **dataset_kwargs)

        np.random.seed(42)
        train_size = int(n_vectors * (1 - test_size))
        train_vecs = train_vecs[np.random.choice(train_vecs.shape[0], train_size)]
        test_vecs = test_vecs[
            np.random.choice(test_vecs.shape[0], n_vectors - train_size)
        ]

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return train_vecs, test_vecs, metadata


def get_metadata(
    synthesized: bool = False,
    train_vecs: np.ndarray | None = None,
    test_vecs: np.ndarray | None = None,
    dataset_name: str | None = None,
    **metadata_kwargs,
) -> tuple[Metadata, Metadata]:
    """Load or synthesize metadata for the train and test splits of a dataset

    Parameters
    ----------
    synthesized : bool, optional
        Whether to use synthesized metadata, by default False
    train_vecs : np.ndarray | None, optional
        Train split of the dataset, by default None
        Only used if synthesized is True
    test_vecs : np.ndarray | None, optional
        Test split of the dataset, by default None
        Only used if synthesized is True
    dataset_name : str | None, optional
        Name of the dataset, by default None
        Only used if synthesized is False

    Returns
    -------
    tuple[Metadata, Metadata]
        Metadata for the train and test splits, respectively
    """

    if not synthesized:
        if dataset_name is None:
            raise ValueError("dataset_name must be specified for real metadata")

        if dataset_name == "arxiv":
            from dataset.arxiv_dataset import load_arxiv_dataset_mds

            train_mds, test_mds = load_arxiv_dataset_mds(**metadata_kwargs)
            return train_mds, test_mds

        elif dataset_name == "yfcc100m":
            from dataset.yfcc100m_dataset import load_yfcc_dataset_mds

            train_mds, test_mds = load_yfcc_dataset_mds(**metadata_kwargs)
            return train_mds, test_mds

        elif dataset_name.startswith("randperm"):
            from dataset.randperm_dataset import load_randperm_dataset_mds

            train_mds, test_mds = load_randperm_dataset_mds(**metadata_kwargs)
            return train_mds, test_mds

        else:
            raise NotImplementedError(
                f"No metadata available for dataset {dataset_name}"
            )
    else:
        raise NotImplementedError("Synthesized metadata not implemented yet")
