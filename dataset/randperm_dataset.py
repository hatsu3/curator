import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random

from benchmark.config import DatasetConfig
from dataset import DATA_DIR

RANDPERM_DIR = Path(DATA_DIR) / "randperm"
RANDPERM_DIR.mkdir(parents=True, exist_ok=True)


def load_randperm_dataset_mds(
    n_vecs: int, n_tenants: int, access_matrix_density: float, test_size: float, seed=42
):
    cache_path = (
        RANDPERM_DIR
        / f"n{n_vecs}_t{n_tenants}_d{access_matrix_density}_tsize{test_size}_seed{seed}.pkl"
    )
    if cache_path.exists():
        print(f"Loading metadata from {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    access_matrix = sparse_random(
        n_vecs,
        n_tenants,
        density=access_matrix_density,
        format="csr",
        random_state=seed,
    )
    assert isinstance(access_matrix, csr_matrix)
    metadata = [access_matrix[row].nonzero()[1].tolist() for row in range(n_vecs)]
    train_size = int(n_vecs * (1 - test_size))
    train_mds = metadata[:train_size]
    test_mds = metadata[train_size:]

    print(f"Saving metadata to {cache_path}")
    with open(cache_path, "wb") as f:
        pkl.dump((train_mds, test_mds), f)

    return train_mds, test_mds


@dataclass
class RandPermDatasetConfig(DatasetConfig):
    dataset_name: str = field(default="randperm", init=False, repr=True)

    def validate_params(self):
        assert "real_ds_name" in self.dataset_params
        assert self.metadata_params.keys() <= {
            "n_vecs",
            "n_tenants",
            "access_matrix_density",
            "test_size",
            "seed",
        }


if __name__ == "__main__":
    mds = load_randperm_dataset_mds(30, 30, 0.2, 0.1)
    print(mds)
