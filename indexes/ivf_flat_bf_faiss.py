from typing import Any

import faiss
import numpy as np

from indexes.base import Index
from dataset import Metadata


class IVFFlatMultiTenantBFFaiss(Index):
    """ IVF-Flat index augmented with Bloom filters """

    def __init__(
        self,
        d: int,
        nlist: int,
        bf_capacity: int = 100,
        bf_error_rate: float = 0.001,
        gamma: float = 5.0,
    ):
        """Initialize an IVFFlat index.

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
        gamma : float, optional
            The scaling factor of the candidate set during knn search.
            Larger values of gamma will lead to more accurate results but slower search.
        """
        super().__init__()

        self.d = d
        self.nlist = nlist
        self.bf_capacity = bf_capacity
        self.bf_error_rate = bf_error_rate
        self.gamma = gamma

        self.quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.MultiTenantIndexIVFFlatBF(
            self.quantizer,
            self.d,
            self.nlist,
            faiss.METRIC_L2,
            self.bf_capacity,
            self.bf_error_rate,
            self.gamma,
        )

        self.label_to_tenants: dict[int, set[int]] = {}

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "nlist": self.nlist,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {"gamma": self.gamma}

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        self.gamma = params["gamma"]

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


if __name__ == "__main__":
    print("Testing IVFFlatMultiTenantBFFaiss...")
    index = IVFFlatMultiTenantBFFaiss(10, 10)
    index.train(np.random.random((400, 10)))
    index.create(np.random.rand(10), 0, 0)
    index.create(np.random.rand(10), 1, 0)
    index.create(np.random.rand(10), 2, 1)
    index.grant_access(2, 2)
    res = index.query(np.random.rand(10), 2, 0)
    print(res)
