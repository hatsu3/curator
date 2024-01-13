from itertools import product

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config
from indexes.ivf_flat_sepidx_faiss import IVFFlatMultiTenantSepIndexFaiss


def exp_ivf_faiss_sep_index(
    nlist_space=[10, 20, 30, 40],
    nprobe_space=[2, 4, 6, 8],
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"output/ivf_faiss_sep_index_{dataset_key}.csv"

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=IVFFlatMultiTenantSepIndexFaiss,
            index_params={
                "d": dim,
                "nlist": nlist,
            },
            search_params={
                "nprobe": nprobe,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        for nlist, nprobe in product(nlist_space, nprobe_space)
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
    fire.Fire(exp_ivf_faiss_sep_index)
