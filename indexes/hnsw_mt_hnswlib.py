from typing import Any

import faiss
import numpy as np

from indexes.base import Index
from dataset import Metadata


class HNSWMultiTenantHnswlib(Index):
    """ HNSW index with metadata filtering """

    def __init__(
        self,
        construction_ef: int = 100,
        search_ef: int = 10,
        m: int = 16,
        num_threads: int = 1,
        max_elements: int = 2000000,
    ) -> None:
        super().__init__()

        self.metric = "l2"
        self.construction_ef = construction_ef
        self.search_ef = search_ef
        self.m = m
        self.num_threads = num_threads
        self.max_elements = max_elements

        self.index: faiss.MultiTenantIndexHNSW | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {
            "construction_ef": self.construction_ef,
            "m": self.m,
            "num_threads": self.num_threads,
            "max_elements": self.max_elements,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "search_ef": self.search_ef,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "search_ef" in params:
            self.search_ef = params["search_ef"]

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        raise NotImplementedError("hnswlib does not require training")

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        if self.index is None:
            self.index = faiss.MultiTenantIndexHNSW(
                len(x), self.m, self.construction_ef, self.search_ef, self.max_elements
            )

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
        _, top_ids = self.index.search(x[None], k, tenant_id)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        _, top_ids = self.index.search(X, k, tenant_id)  # type: ignore
        return top_ids.tolist()


if __name__ == "__main__":
    index = HNSWMultiTenantHnswlib()
    index.create(np.random.rand(10), label=0, tenant_id=0)
    index.create(np.random.rand(10), label=1, tenant_id=1)
    index.grant_access(label=0, tenant_id=1)
    index.grant_access(label=1, tenant_id=0)
    index.grant_access(label=1, tenant_id=2)
    print(index.query(np.random.rand(10), k=2, tenant_id=0))
    index.delete_vector(label=1, tenant_id=1)
    print(index.query(np.random.rand(10), k=1, tenant_id=0))
