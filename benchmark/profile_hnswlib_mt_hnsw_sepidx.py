from itertools import product

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config
from indexes.hnsw_sepidx_hnswlib import HNSWMultiTenantSepIndexHnswlib


def exp_hnsw_sepidx_hnswlib(
    construction_ef_space=[16, 32, 64],
    search_ef_space=[16, 32, 64],
    m_space=[32, 48, 64],
    max_elements=2000000,
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"output/hnsw_sepidx_hnswlib_{dataset_key}.csv"

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=HNSWMultiTenantSepIndexHnswlib,
            index_params={
                "construction_ef": construction_ef,
                "m": m,
                "max_elements": max_elements,
            },
            search_params={
                "search_ef": search_ef,
            },
            train_params=None,
        )
        for construction_ef, search_ef, m in product(
            construction_ef_space, search_ef_space, m_space
        )
    ]

    profiler = IndexProfiler(multi_tenant=True)
    results = profiler.batch_profile(
        index_configs, [dataset_config], num_runs=num_runs, timeout=timeout
    )

    if output_path is not None:
        df = pd.DataFrame(
            [
                {
                    **config.index_params,
                    **config.search_params,
                    **res,
                }
                for res, config in zip(results, index_configs)
            ]
        )
        df.to_csv(output_path, index=False)

    return results


if __name__ == "__main__":
    fire.Fire(exp_hnsw_sepidx_hnswlib)
