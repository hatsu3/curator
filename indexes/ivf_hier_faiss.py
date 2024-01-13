from typing import Any

import faiss
import numpy as np

from indexes.base import Index
from dataset import Metadata


class IVFFlatMultiTenantBFHierFaiss(Index):
    """ Curator index """

    def __init__(
        self,
        d: int,
        nlist: int,
        bf_capacity: int = 1000,
        bf_error_rate: float = 0.001,
        gamma1: float = 16.0,
        gamma2: float = 256.0,
        max_sl_size: int = 128,
    ):
        """Initialize Curator index.

        Parameters
        ----------
        d : int
            Dimensionality of the vectors.
        nlist : int
            Number of cells/buckets in the inverted file.
        bf_capacity : int, optional
            The capacity of the Bloom filter.
        bf_error_rate : float, optional
            The error rate of the Bloom filter.
        gamma1, gamma2 : float, optional
            The scaling factor of the candidate set during knn search.
            Larger values of gamma will lead to more accurate results but slower search.
        """
        super().__init__()

        self.d = d
        self.nlist = nlist
        self.bf_capacity = bf_capacity
        self.bf_error_rate = bf_error_rate
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_sl_size = max_sl_size

        self.quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.MultiTenantIndexIVFHierarchical(
            self.quantizer,
            self.d,
            self.nlist,
            faiss.METRIC_L2,
            self.bf_capacity,
            self.bf_error_rate,
            self.gamma1,
            self.gamma2,
            self.max_sl_size,
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "nlist": self.nlist,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
            "max_sl_size": self.max_sl_size,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "gamma1": self.gamma1,
            "gamma2": self.gamma2,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "gamma1" in params:
            self.gamma1 = params["gamma1"]
        if "gamma2" in params:
            self.gamma2 = params["gamma2"]

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        self.index.train(X, 0)  # type: ignore

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        self.index.add_vector_with_ids(x[None], [label], tenant_id)  # type: ignore

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)  # type: ignore

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("Use delete_vector instead")

    def delete_vector(self, label: int, tenant_id: int) -> None:
        self.index.remove_vector(label, tenant_id)  # type: ignore

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id)  # type: ignore

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, tenant_id)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        top_dists, top_ids = self.index.search(X, k, tenant_id)  # type: ignore
        return top_ids.tolist()


if __name__ == "__main__":
    print("Testing IVFFlatMultiTenantBFHierFaiss...")
    index = IVFFlatMultiTenantBFHierFaiss(10, 10)
    index.train(np.random.random((400, 10)))
    index.create(np.random.rand(10), 0, 0)
    index.create(np.random.rand(10), 1, 0)
    index.create(np.random.rand(10), 2, 1)
    index.grant_access(2, 2)
    res = index.query(np.random.rand(10), 2, 0)
    print(res)
