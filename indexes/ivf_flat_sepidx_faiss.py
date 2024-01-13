from typing import Any

import faiss
import numpy as np

from indexes.base import Index
from dataset import Metadata


class IVFFlatMultiTenantSepIndexFaiss(Index):
    """ IVF-Flat index with per-tenant indexing """

    def __init__(self, d: int, nlist: int, nprobe: int = 5) -> None:
        """ Initialize an IVFFlat index.
        
        Parameters
        ----------
        d : int
            The dimension of the vectors.
        nlist : int
            The number of cells in the inverted file.
        nprobe : int, optional
            The number of cells to search during query, by default 5
        """
        super().__init__()

        # same index parameters for all tenants
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe

        self.quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.MultiTenantIndexIVFFlatSep(self.quantizer, d, nlist)

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
        # train the index for each tenant
        assert tenant_ids is not None

        all_tenants = set()
        for tids in tenant_ids:
            all_tenants.update(tids)  # type: ignore

        for tid in all_tenants:
            # select the vectors accessible to the tenant
            X_tenant = X[[tid in tids for tids in tenant_ids]]
            self.index.train(X_tenant, int(tid))  # type: ignore

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        self.index.add_vector_with_ids(x[None], [label], tenant_id)  # type: ignore

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("Use delete_vector instead")

    def delete_vector(self, label: int, tenant_id: int) -> None:
        self.index.remove_vector(label, tenant_id)

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id)

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        if tenant_id is None:
            raise ValueError("Please specify the tenant id")

        params = faiss.SearchParametersIVF(nprobe=self.nprobe)  # type: ignore
        top_dists, top_ids = self.index.search(x[None], k, tenant_id, params=params)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1) -> list[list[int]]:
        params = faiss.SearchParametersIVF(nprobe=self.nprobe)  # type: ignore
        top_dists, top_ids = self.index.search(X, k, tenant_id, params=params)  # type: ignore
        return top_ids.tolist()
