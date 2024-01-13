import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from benchmark.config import DatasetConfig
from dataset import DATA_DIR

YFCC100M_DIR = Path(DATA_DIR) / "yfcc100m"
YFCC100M_DIR.mkdir(parents=True, exist_ok=True)


def read_sparse_matrix_fields(fname):
    """read the fields of a CSR matrix without instanciating it"""
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype="int32", count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype="float32", count=nnz)
        return data, indices, indptr, ncol


def read_sparse_matrix(fname):
    """read a CSR matrix in spmat format, optionally mmapping it instead"""
    data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))


def load_vecs(path, dtype=np.uint8):
    n, d = map(int, np.fromfile(path, dtype="uint32", count=2))
    vecs = np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))
    vecs = np.ascontiguousarray(vecs)
    return vecs


def load_raw_yfcc_dataset(split: str = "train"):
    """
    Please download the dataset from
    https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.10M.u8bin and
    https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.metadata.10M.spmat
    and place them in the directory `data/yfcc100m` (YFCC100M_DIR)
    """

    assert split == "train", "only train split is supported for now"
    vecs_path = YFCC100M_DIR / "base.10M.u8bin"
    mds_path = YFCC100M_DIR / "base.metadata.10M.spmat"

    vecs = load_vecs(vecs_path)
    mds = read_sparse_matrix(mds_path)

    return vecs, mds


def top_n_m_submatrix(matrix, n, m):
    """
    Returns a sub-matrix from a CSR matrix that contains the top-n rows and top-m columns
    with the most non-zero elements.
    """
    assert isinstance(matrix, csr_matrix)
    nonzeros_per_row = np.diff(matrix.indptr)
    nonzeros_per_column = np.diff(matrix.tocsc().indptr)

    top_n_rows = np.argsort(nonzeros_per_row)[-n:]
    top_m_columns = np.argsort(nonzeros_per_column)[-m:]

    submatrix = matrix[top_n_rows, :][:, top_m_columns]
    return submatrix, top_n_rows


def subsample_yfcc_dataset(
    n_vectors: int = 1_000_000,
    n_labels: int = 1000,
    save_dir: str | Path = YFCC100M_DIR,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    prefix = f"yfcc_subsampled_nvec_{n_vectors}_nlabel_{n_labels}"
    mds_path = save_dir / f"{prefix}_mds.pkl"
    vecs_path = save_dir / f"{prefix}_vecs.npy"

    vecs, mds = load_raw_yfcc_dataset()
    mds_sampled, vecs_ids_sampled = top_n_m_submatrix(mds, n_vectors, n_labels)
    vecs_sampled = vecs[vecs_ids_sampled]
    vecs_sampled = vecs_sampled.astype(np.float32) / 255.0 - 0.5

    # convert metadata in sparse matrix format to list of lists
    assert isinstance(mds_sampled, csr_matrix)
    indptr = mds_sampled.indptr
    mds_sampled_lst = [
        mds_sampled.indices[indptr[i] : indptr[i + 1]].tolist()
        for i in range(mds_sampled.shape[0])
    ]

    with open(mds_path, "wb") as f:
        pkl.dump(mds_sampled_lst, f)

    np.save(vecs_path, vecs_sampled)


def train_test_split(dataset_size: int, test_size: float = 0.2, seed: int = 42):
    np.random.seed(seed=seed)
    train_size = int(dataset_size * (1 - test_size))
    shuffled_idxs = np.random.permutation(dataset_size)
    train_indices = shuffled_idxs[:train_size]
    test_indices = shuffled_idxs[train_size:]

    return train_indices, test_indices


def load_yfcc_dataset_vecs(
    data_path: str
    | Path = YFCC100M_DIR / "yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy",
    test_size: float = 0.2,
    seed: int = 42,
):
    data_path = Path(data_path)
    assert data_path.exists(), f"{data_path} does not exist"

    vecs = np.load(data_path)
    train_indices, test_indices = train_test_split(len(vecs), test_size, seed)
    train_vecs = vecs[train_indices]
    test_vecs = vecs[test_indices]

    return train_vecs, test_vecs


def load_yfcc_dataset_mds(
    data_path: str
    | Path = YFCC100M_DIR / "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl",
    test_size: float = 0.2,
    seed: int = 42,
):
    data_path = Path(data_path)
    assert data_path.exists(), f"{data_path} does not exist"

    with open(data_path, "rb") as f:
        mds = pkl.load(f)

    train_indices, test_indices = train_test_split(len(mds), test_size, seed)
    train_mds = [mds[i] for i in train_indices]
    test_mds = [mds[i] for i in test_indices]

    return train_mds, test_mds


@dataclass
class YFCC100MDatasetConfig(DatasetConfig):
    dataset_name: str = field(default="yfcc100m", init=False, repr=True)

    def validate_params(self):
        assert self.dataset_params.keys() <= {
            "data_path",
            "test_size",
            "seed",
        }
        assert self.metadata_params.keys() <= {
            "data_path",
            "test_size",
            "seed",
        }


if __name__ == "__main__":
    subsample_yfcc_dataset(n_vectors=1_000_000, n_labels=1000)
    train_vecs, test_vecs = load_yfcc_dataset_vecs(
        YFCC100M_DIR / "yfcc_subsampled_nvec_1000000_nlabel_1000_vecs.npy"
    )
    train_mds, test_mds = load_yfcc_dataset_mds(
        YFCC100M_DIR / "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl"
    )
