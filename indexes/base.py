from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from dataset import Metadata


class Index(ABC):
    """Base class for all indexes."""

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return the index parameters.
        These parameters are set at initialization time and cannot be changed.
        """
        ...

    @property
    @abstractmethod
    def search_params(self) -> dict[str, Any]:
        """Return the search parameters.
        These parameters can be changed at any time.
        """
        ...

    @search_params.setter
    @abstractmethod
    def search_params(self, params: dict[str, Any]) -> None:
        ...

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        """Train the index.

        Parameters
        ----------
        X : np.ndarray
            The training vectors.
        tenant_ids : Optional[Metadata], optional
            The metadata of the training vectors.
        """
        return

    def insert(
        self, x: np.ndarray, label: int, tenant_ids: list[int] | None = None
    ) -> None:
        """Insert a vector into the index for the specified tenants.

        Parameters
        ----------
        x : np.ndarray
            The vector to insert.
        label : int
            The external label of the vector, which is used by the user to identify it.
        tenant_ids : Optional[list[int]], optional
            The list of tenant IDs that have access to the vector.
            If None, the vector is accessible to all tenants (no multi-tenancy).
            This is possible only if multi-tenancy is disabled or it's a query from the admin tenant.
        """
        raise NotImplementedError

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        """Insert a new vector into the index.

        Parameters
        ----------
        x : np.ndarray
            The vector to insert.
        label : int
            The external label of the vector, which is used by the user to identify it.
        tenant_id : int
            ID of the tenant that is creating the vector
        """
        raise NotImplementedError

    def grant_access(self, label: int, tenant_id: int) -> None:
        """Grant access to a vector to a tenant.

        Parameters
        ----------
        label : int
            The external label of the vector.
        tenant_id : int
            The ID of the tenant to grant access to.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, label: int, tenant_id: int | None = None) -> None:
        """Delete a vector from the index.

        Parameters
        ----------
        label : int
            The external label of the vector to delete.
        tenant_id : Optional[int], optional
            The ID of the querying tenant.
            If None, the corresponding vector is deleted from all tenants.
            This is possible only if multi-tenancy is disabled or it's a query from the admin tenant.
            Otherwise, the vector is deleted only from the specified tenant.
        """
        ...

    def delete_vector(self, label: int, tenant_id: int) -> None:
        """Delete a vector from the index.

        Parameters
        ----------
        label : int
            The external label of the vector to delete.
        tenant_id : int
            The ID of the querying tenant.
        """
        raise NotImplementedError

    def revoke_access(self, label: int, tenant_id: int) -> None:
        """Revoke access to a vector from a tenant.

        Parameters
        ----------
        label : int
            The external label of the vector.
        tenant_id : int
            The ID of the tenant to revoke access from.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        """Query the index for the k nearest neighbors of a vector.

        Parameters
        ----------
        x : np.ndarray
            The query vector.
        k : int
            The number of nearest neighbors to return.
        tenant_id : Optional[int], optional
            The ID of the querying tenant.
            If None, the query is performed by the admin/only tenant and all vectors are considered.

        Returns
        -------
        list[int]
            The list of external labels of the k nearest neighbors.
            If there are less than k neighbors, the list is **not** padded.
        """
        ...

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        """Query the index for the k nearest neighbors of multiple vectors.

        Parameters
        ----------
        X : np.ndarray
            The query vectors.
        k : int
            The number of nearest neighbors to return.
        tenant_id : Optional[int], optional
            The ID of the querying tenant.
            If None, the query is performed by the admin/only tenant and all vectors are considered.
        num_threads : int, optional
            The number of threads to use for the query (default: 1).

        Returns
        -------
        list[list[int]]
            The list of list of external labels of the k nearest neighbors.
            If there are less than k neighbors, the list is **not** padded.
        """
        raise NotImplementedError

    def save(self, path: Path | str) -> None:
        """Save the index to disk.

        Parameters
        ----------
        path : str
            The path where to save the index.
        """
        raise NotImplementedError

    def load(self, path: Path | str) -> None:
        """Load the index from disk.

        Parameters
        ----------
        path : str
            The path from where to load the index.
        """
        raise NotImplementedError

    def shrink_to_fit(self) -> None:
        """Shrink the index to fit the current number of elements."""
        raise NotImplementedError
