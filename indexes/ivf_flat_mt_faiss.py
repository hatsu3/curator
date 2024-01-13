from typing import Any

import faiss
import numpy as np

from indexes.base import Index
from dataset import Metadata


class IVFFlatMultiTenantFaiss(Index):
    """ IVF-Flat index with metadata filtering """

    def __init__(self, d: int, nlist: int, nprobe: int = 1) -> None:
        super().__init__()
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe

        self.quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.MultiTenantIndexIVFFlat(
            self.quantizer,
            self.d,
            self.nlist,
            faiss.METRIC_L2,
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "nlist": self.nlist,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "nprobe": self.nprobe,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "nprobe" in params:
            self.nprobe = params["nprobe"]

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
        self.index.remove_vector(label, tenant_id) # type: ignore

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id) # type: ignore

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        params = faiss.SearchParametersIVF(nprobe=self.nprobe)  # type: ignore
        top_dists, top_ids = self.index.search(x[None], k, tenant_id, params=params)  # type: ignore
        return top_ids[0].tolist()
    
    def batch_query(self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1) -> list[list[int]]:
        params = faiss.SearchParametersIVF(nprobe=self.nprobe)  # type: ignore
        top_dists, top_ids = self.index.search(X, k, tenant_id, params=params)  # type: ignore
        return top_ids.tolist()


if __name__ == "__main__":
    print("Testing IVFFlatMultiTenantFaiss...")
    index = IVFFlatMultiTenantFaiss(10, 10)
    index.train(np.random.random((400, 10)))
    index.create(np.random.rand(10), 0, 0)
    index.create(np.random.rand(10), 1, 0)
    index.create(np.random.rand(10), 2, 1)
    index.grant_access(2, 2)
    res = index.query(np.random.rand(10), 2, 0)
    print(res)
