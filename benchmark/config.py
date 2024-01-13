import json
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from indexes.base import Index


@dataclass
class Config(ABC):
    def __post_init__(self):
        self.validate_params()

    def validate_params(self):
        pass

    @property
    def json(self) -> dict:
        ...

    def __repr__(self) -> str:
        return json.dumps(self.json, indent=4)


@dataclass
class IndexConfig(Config):
    index_cls: Type[Index]
    index_params: dict
    search_params: dict
    train_params: dict | None = None

    @property
    def json(self) -> dict:
        return {
            "index_cls": self.index_cls.__name__,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "train_params": self.train_params,
        }


@dataclass
class DatasetConfig(Config):
    dataset_name: str
    dataset_params: dict
    synthesize_metadata: bool
    metadata_params: dict

    @property
    def json(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "dataset_params": {
                k: v if not isinstance(v, Path) else str(v)
                for k, v in self.dataset_params.items()
            },
            "synthesize_metadata": self.synthesize_metadata,
            "metadata_params": {
                k: v if not isinstance(v, Path) else str(v)
                for k, v in self.metadata_params.items()
            },
        }

    def _validate_metadata_params(self):
        pass
