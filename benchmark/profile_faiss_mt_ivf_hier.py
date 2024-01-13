from itertools import product

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss


def exp_ivf_hier_faiss(
    nlist_space=[4, 8, 16, 32],
    gamma1_space=[8.0, 16.0, 24.0],
    gamma2_space=[128.0, 256.0, 512.0],
    max_sl_size_space=[32, 64, 128, 256],
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"output/ivf_hier_faiss_{dataset_key}.csv"

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=IVFFlatMultiTenantBFHierFaiss,
            index_params={
                "d": dim,
                "nlist": nlist,
                "bf_capacity": 1000,
                "bf_error_rate": 0.01,
                "max_sl_size": max_sl_size,
            },
            search_params={
                "gamma1": gamma1,
                "gamma2": gamma2,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        for nlist, gamma1, gamma2, max_sl_size in product(
            nlist_space, gamma1_space, gamma2_space, max_sl_size_space
        )
    ]

    profiler = IndexProfiler(multi_tenant=True)
    results = profiler.batch_profile(index_configs, [dataset_config], num_runs=num_runs, timeout=timeout)

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
    fire.Fire(exp_ivf_hier_faiss)
