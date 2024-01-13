import logging
from itertools import product
from pathlib import Path

import fire

from run_docker import run_docker

# pylint: disable=W0105
# pylint: disable=C0302

""" Parameter sweep """


def run_curator_exp(
    nlist: int,
    gamma1: int,
    gamma2: int,
    max_sl_size: int,
    cpu_limit: str,
    log_path: str | None = None,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_hier",
            "--nlist_space",
            f"[{nlist}]",
            "--gamma1_space",
            f"[{gamma1}]",
            "--gamma2_space",
            f"[{gamma2}]",
            "--max_sl_size_space",
            f"[{max_sl_size}]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_ivf_exp(
    nlist: int,
    nprobe: int,
    cpu_limit: str,
    log_path: str | None = None,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf",
            "--nlist_space",
            f"[{nlist}]",
            "--nprobe_space",
            f"[{nprobe}]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_ivf_exp(
    nlist: int,
    nprobe: int,
    cpu_limit: str,
    log_path: str | None = None,
    mem_limit: int = 100_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_sepidx",
            "--nlist_space",
            f"[{nlist}]",
            "--nprobe_space",
            f"[{nprobe}]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_hnsw_exp(
    construction_ef: int,
    search_ef: int,
    m: int,
    cpu_limit: str,
    log_path: str | None = None,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw",
            "--construction_ef_space",
            f"[{construction_ef}]",
            "--search_ef_space",
            f"[{search_ef}]",
            "--m_space",
            f"[{m}]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_hnsw_exp(
    construction_ef: int,
    search_ef: int,
    m: int,
    cpu_limit: str,
    log_path: str | None = None,
    mem_limit: int = 100_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw_sepidx",
            "--construction_ef_space",
            f"[{construction_ef}]",
            "--search_ef_space",
            f"[{search_ef}]",
            "--m_space",
            f"[{m}]",
            "--max_elements",
            "400000",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_curator_param_sweep(
    nlist_space: list[int],
    gamma1_space: list[int],
    gamma2_space: list[int],
    max_sl_size_space: list[int],
    cpu_limit: str,
    log_dir: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    for nlist, gamma1, gamma2, max_sl_size in product(
        nlist_space, gamma1_space, gamma2_space, max_sl_size_space
    ):
        log_path = (
            log_dir_path
            / f"nlist={nlist}-gamma1={gamma1}-gamma2={gamma2}-max_sl_size={max_sl_size}.log"
        ).absolute()

        run_curator_exp(
            nlist=nlist,
            gamma1=gamma1,
            gamma2=gamma2,
            max_sl_size=max_sl_size,
            cpu_limit=cpu_limit,
            log_path=str(log_path),
            mem_limit=mem_limit,
            num_runs=num_runs,
            timeout=timeout,
        )


def run_shared_ivf_param_sweep(
    nlist_space: list[int],
    nprobe_space: list[int],
    cpu_limit: str,
    log_dir: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    for nlist, nprobe in product(nlist_space, nprobe_space):
        log_path = (log_dir_path / f"nlist={nlist}-nprobe={nprobe}.log").absolute()
        run_shared_ivf_exp(
            nlist=nlist,
            nprobe=nprobe,
            cpu_limit=cpu_limit,
            log_path=str(log_path),
            mem_limit=mem_limit,
            num_runs=num_runs,
            timeout=timeout,
        )


def run_separate_ivf_param_sweep(
    nlist_space: list[int],
    nprobe_space: list[int],
    cpu_limit: str,
    log_dir: str,
    mem_limit: int = 100_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    for nlist, nprobe in product(nlist_space, nprobe_space):
        log_path = (log_dir_path / f"nlist={nlist}-nprobe={nprobe}.log").absolute()
        run_separate_ivf_exp(
            nlist=nlist,
            nprobe=nprobe,
            cpu_limit=cpu_limit,
            log_path=str(log_path),
            mem_limit=mem_limit,
            num_runs=num_runs,
            timeout=timeout,
        )


def run_shared_hnsw_param_sweep(
    construction_ef_space: list[int],
    search_ef_space: list[int],
    m_space: list[int],
    cpu_limit: str,
    log_dir: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    for construction_ef, search_ef, m in product(
        construction_ef_space, search_ef_space, m_space
    ):
        log_path = (
            log_dir_path
            / f"construction_ef={construction_ef}-search_ef={search_ef}-m={m}.log"
        ).absolute()

        run_shared_hnsw_exp(
            construction_ef=construction_ef,
            search_ef=search_ef,
            m=m,
            cpu_limit=cpu_limit,
            log_path=str(log_path),
            mem_limit=mem_limit,
            num_runs=num_runs,
            timeout=timeout,
        )


def run_separate_hnsw_param_sweep(
    construction_ef_space: list[int],
    search_ef_space: list[int],
    m_space: list[int],
    cpu_limit: str,
    log_dir: str,
    mem_limit: int = 100_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    for construction_ef, search_ef, m in product(
        construction_ef_space, search_ef_space, m_space
    ):
        log_path = (
            log_dir_path
            / f"construction_ef={construction_ef}-search_ef={search_ef}-m={m}.log"
        ).absolute()

        run_separate_hnsw_exp(
            construction_ef=construction_ef,
            search_ef=search_ef,
            m=m,
            cpu_limit=cpu_limit,
            log_path=str(log_path),
            mem_limit=mem_limit,
            num_runs=num_runs,
            timeout=timeout,
        )


""" Overall results """


def run_curator_overall_exp(
    dataset,
    cpu_limit: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_hier",
                "--nlist_space",
                "[48]",
                "--gamma1_space",
                "[2]",
                "--gamma2_space",
                "[384]",
                "--max_sl_size_space",
                "[256]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_hier",
                "--nlist_space",
                "[32]",
                "--gamma1_space",
                "[2]",
                "--gamma2_space",
                "[128]",
                "--max_sl_size_space",
                "[256]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


def run_shared_ivf_overall_exp(
    dataset,
    cpu_limit: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf",
                "--nlist_space",
                "[400]",
                "--nprobe_space",
                "[48]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf",
                "--nlist_space",
                "[800]",
                "--nprobe_space",
                "[32]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


def run_separate_ivf_overall_exp(
    dataset,
    cpu_limit: str,
    mem_limit: int = 100_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_sepidx",
                "--nlist_space",
                "[320]",
                "--nprobe_space",
                "[24]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_sepidx",
                "--nlist_space",
                "[320]",
                "--nprobe_space",
                "[16]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


def run_shared_hnsw_overall_exp(
    dataset,
    cpu_limit: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_hnswlib_mt_hnsw",
                "--construction_ef_space",
                "[32]",
                "--search_ef_space",
                "[32]",
                "--m_space",
                "[32]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_hnswlib_mt_hnsw",
                "--construction_ef_space",
                "[32]",
                "--search_ef_space",
                "[16]",
                "--m_space",
                "[32]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


def run_separate_hnsw_overall_exp(
    dataset,
    cpu_limit: str,
    mem_limit: int = 100_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_hnswlib_mt_hnsw_sepidx",
                "--construction_ef_space",
                "[64]",
                "--search_ef_space",
                "[96]",
                "--m_space",
                "[32]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_hnswlib_mt_hnsw_sepidx",
                "--construction_ef_space",
                "[32]",
                "--search_ef_space",
                "[16]",
                "--m_space",
                "[48]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


""" Ablation study """


def run_curator_flat_search_overall_exp(
    dataset,
    cpu_limit: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "FLAT_SEARCH=1",
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_hier",
                "--nlist_space",
                "[48]",
                "--gamma1_space",
                "[2]",
                "--gamma2_space",
                "[384]",
                "--max_sl_size_space",
                "[256]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "FLAT_SEARCH=1",
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_hier",
                "--nlist_space",
                "[32]",
                "--gamma1_space",
                "[2]",
                "--gamma2_space",
                "[128]",
                "--max_sl_size_space",
                "[256]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


def run_shared_ivf_bf_overall_exp(
    dataset,
    gamma: int,
    cpu_limit: str,
    mem_limit: int = 20_000_000_000,
    num_runs: int = 1,
    timeout: int = 600,
):
    assert dataset in ["arxiv-large", "yfcc100m"]
    logging.basicConfig(level=logging.INFO)

    if dataset == "arxiv-large":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_bf",
                "--nlist_space",
                "[400]",
                "--gamma_space",
                f"[{gamma}]",
                "--dataset_key",
                "arxiv-large-10",
                "--test_size",
                "0.005",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )
    elif dataset == "yfcc100m":
        run_docker(
            cmd=[
                "python",
                "-m",
                "benchmark.profile_faiss_mt_ivf_bf",
                "--nlist_space",
                "[800]",
                "--gamma_space",
                f"[{gamma}]",
                "--dataset_key",
                "yfcc100m",
                "--test_size",
                "0.01",
                "--num_runs",
                str(num_runs),
                "--timeout",
                str(timeout),
            ],
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
        )


""" Multi-threaded search """


def run_curator_mt_exp(
    num_threads: int,
    parallel_mode: str,
    cpu_limit: str,
    mem_limit=20_000_000_000,
    num_runs: int = 1,
    timeout: int | None = None,
):
    assert parallel_mode in ["inter", "intra"]
    logging.basicConfig(level=logging.INFO)

    envvars = [f"OMP_NUM_THREADS={num_threads}"]
    if parallel_mode == "inter":
        envvars.append("BATCH_QUERY=1")

    run_docker(
        cmd=[
            *envvars,
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_hier",
            "--nlist_space",
            "[32]",
            "--gamma1_space",
            "[2]",
            "--gamma2_space",
            "[128]",
            "--max_sl_size_space",
            "[256]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_ivf_mt_exp(
    num_threads: int,
    parallel_mode: str,
    cpu_limit: str,
    mem_limit=20_000_000_000,
    num_runs: int = 1,
    timeout: int | None = None,
):
    assert parallel_mode in ["inter", "intra"]
    logging.basicConfig(level=logging.INFO)

    envvars = [f"OMP_NUM_THREADS={num_threads}"]
    if parallel_mode == "inter":
        envvars.append("BATCH_QUERY=1")

    run_docker(
        cmd=[
            *envvars,
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf",
            "--nlist_space",
            "[800]",
            "--nprobe_space",
            "[32]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_ivf_mt_exp(
    num_threads: int,
    parallel_mode: str,
    cpu_limit: str,
    mem_limit=20_000_000_000,
    num_runs: int = 1,
    timeout: int | None = None,
):
    assert parallel_mode in ["inter", "intra"]
    logging.basicConfig(level=logging.INFO)

    envvars = [f"OMP_NUM_THREADS={num_threads}"]
    if parallel_mode == "inter":
        envvars.append("BATCH_QUERY=1")

    run_docker(
        cmd=[
            *envvars,
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_sepidx",
            "--nlist_space",
            "[320]",
            "--nprobe_space",
            "[16]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_hnsw_mt_exp(
    num_threads: int,
    cpu_limit: str,
    mem_limit=20_000_000_000,
    num_runs: int = 1,
    timeout: int | None = None,
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "BATCH_QUERY=1",
            f"OMP_NUM_THREADS={num_threads}",
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw",
            "--construction_ef_space",
            "[32]",
            "--search_ef_space",
            "[16]",
            "--m_space",
            "[32]",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_hnsw_mt_exp(
    num_threads: int,
    cpu_limit: str,
    mem_limit=100_000_000_000,
    num_runs: int = 1,
    timeout: int | None = None,
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "BATCH_QUERY=1",
            f"OMP_NUM_THREADS={num_threads}",
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw_sepidx",
            "--construction_ef_space",
            "[32]",
            "--search_ef_space",
            "[16]",
            "--m_space",
            "[48]",
            "--max_elements",
            "1000000",
            "--dataset_key",
            "yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            str(num_runs),
            "--timeout",
            str(timeout),
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


""" Scalability """


def run_curator_scalability_exp(density: str, cpu_limit: str, mem_limit=20_000_000_000):
    logging.basicConfig(level=logging.INFO)

    num_vectors = int(10000 / float(density))

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_hier",
            "--nlist_space",
            "[32]",
            "--gamma1_space",
            "[2]",
            "--gamma2_space",
            "[64]",
            "--max_sl_size_space",
            "[256]",
            "--dataset_key",
            f"randperm-v{num_vectors}-t1000-d{density}-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_ivf_scalability_exp(
    density: str, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    num_vectors = int(10000 / float(density))

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf",
            "--nlist_space",
            "[800]",
            "--nprobe_space",
            "[32]",
            "--dataset_key",
            f"randperm-v{num_vectors}-t1000-d{density}-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_ivf_scalability_exp(
    density: str, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    num_vectors = int(10000 / float(density))

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_sepidx",
            "--nlist_space",
            "[320]",
            "--nprobe_space",
            "[16]",
            "--dataset_key",
            f"randperm-v{num_vectors}-t1000-d{density}-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_hnsw_scalability_exp(
    density: str, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    num_vectors = int(10000 / float(density))

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw",
            "--construction_ef_space",
            "[32]",
            "--search_ef_space",
            "[16]",
            "--m_space",
            "[32]",
            "--dataset_key",
            f"randperm-v{num_vectors}-t1000-d{density}-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_hnsw_scalability_exp(
    density: str, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    num_vectors = int(10000 / float(density))

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw_sepidx",
            "--construction_ef_space",
            "[32]",
            "--search_ef_space",
            "[16]",
            "--m_space",
            "[48]",
            "--max_elements",
            "100000",
            "--dataset_key",
            f"randperm-v{num_vectors}-t1000-d{density}-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_curator_scalability_exp2(
    num_tenants: int, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_hier",
            "--nlist_space",
            "[32]",
            "--gamma1_space",
            "[2]",
            "--gamma2_space",
            "[64]",
            "--max_sl_size_space",
            "[256]",
            "--dataset_key",
            f"randperm-v1000000-t{num_tenants}-d0.01-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_ivf_scalability_exp2(
    num_tenants: int, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf",
            "--nlist_space",
            "[800]",
            "--nprobe_space",
            "[32]",
            "--dataset_key",
            f"randperm-v1000000-t{num_tenants}-d0.01-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_ivf_scalability_exp2(
    num_tenants: int, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_faiss_mt_ivf_sepidx",
            "--nlist_space",
            "[320]",
            "--nprobe_space",
            "[16]",
            "--dataset_key",
            f"randperm-v1000000-t{num_tenants}-d0.01-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_shared_hnsw_scalability_exp2(
    num_tenants: int, cpu_limit: str, mem_limit=20_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw",
            "--construction_ef_space",
            "[32]",
            "--search_ef_space",
            "[16]",
            "--m_space",
            "[32]",
            "--dataset_key",
            f"randperm-v1000000-t{num_tenants}-d0.01-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


def run_separate_hnsw_scalability_exp2(
    num_tenants: int, cpu_limit: str, mem_limit=60_000_000_000
):
    logging.basicConfig(level=logging.INFO)

    run_docker(
        cmd=[
            "python",
            "-m",
            "benchmark.profile_hnswlib_mt_hnsw_sepidx",
            "--construction_ef_space",
            "[32]",
            "--search_ef_space",
            "[16]",
            "--m_space",
            "[48]",
            "--max_elements",
            "100000",
            "--dataset_key",
            f"randperm-v1000000-t{num_tenants}-d0.01-yfcc100m",
            "--test_size",
            "0.01",
            "--num_runs",
            "1",
        ],
        cpu_limit=cpu_limit,
        mem_limit=mem_limit,
    )


if __name__ == "__main__":
    fire.Fire()
