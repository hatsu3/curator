import hashlib
import json
import math
import multiprocessing as mp
import pickle as pkl
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from dataset import DATA_DIR, Metadata
from dataset.metrics import Metric

GROUND_TRUTH_DIR = Path(DATA_DIR) / "ground_truth"
SAMPLED_METADATA_DIR = Path(DATA_DIR) / "sampled_metadata"


def compute_ground_truth(
    train_vecs,
    train_mds: Metadata,
    test_vecs,
    test_mds: Metadata,
    k: int = 10,
    metric: Metric = Metric.EUCLIDEAN,
    multi_tenant: bool = True,
    cache_dir = None,
    n_sample_cates: Optional[int] = 10000,
    sample_cates_seed: int = 42,
    use_cuda="auto",
) -> tuple[np.ndarray, set[int]]:
    """Compute the ground truth for k-NN queries.
    """

    if metric != Metric.EUCLIDEAN:
        raise NotImplementedError("Only Euclidean distance is supported")

    train_vecs = np.array(train_vecs)
    test_vecs = np.array(test_vecs)

    cache_dir = Path(cache_dir) if cache_dir is not None else GROUND_TRUTH_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # compute the MD5 hash of train_vecs, train_mds, test_vecs, and test_mds as the cache key
    combined_bytes = (
        train_vecs.tobytes()
        + json.dumps(train_mds, sort_keys=True).encode()
        + test_vecs.tobytes()
        + json.dumps(test_mds, sort_keys=True).encode()
        + json.dumps(
            {"k": k, "metric": metric.value, "mt": multi_tenant}, sort_keys=True
        ).encode()
    )
    combined_md5 = hashlib.md5(combined_bytes).hexdigest()
    cache_path = cache_dir / f"{combined_md5}.npy"

    # all tenant IDs in train_mds
    pkl_path = (
        cache_dir
        / f"train_cates_{combined_md5}_n_sample_{n_sample_cates}_seed_{sample_cates_seed}.pkl"
    )
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            train_cates = pkl.load(f)
    else:
        train_cates = parallel_set_union(train_mds, mp.cpu_count())

        if n_sample_cates is not None:
            np.random.seed(sample_cates_seed)
            n_sample_cates = min(n_sample_cates, len(train_cates))
            train_cates = np.random.choice(
                list(train_cates), n_sample_cates, replace=False
            )
            train_cates = set(train_cates)

        with open(pkl_path, "wb") as f:
            pkl.dump(train_cates, f)

    if Path(cache_path).exists():
        print("Loading ground truth from cache %s..." % cache_path)
        ground_truth = np.load(cache_path)
        return ground_truth, train_cates

    # compute the ground truth using CPU or GPU
    if use_cuda == "auto":
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False

    if use_cuda:
        print("Using GPU to compute the ground truth...")
        ground_truth = compute_ground_truth_cuda(
            train_vecs, train_mds, test_vecs, test_mds, train_cates, k, multi_tenant
        )
    else:
        print("Using CPU to compute the ground truth...")
        ground_truth = compute_ground_truth_cpu(
            train_vecs, train_mds, test_vecs, test_mds, train_cates, k, multi_tenant
        )

    # since we pad -1, the ground truth is not variable-length anymore
    np.save(cache_path, ground_truth)

    return ground_truth, train_cates


def compute_ground_truth_cpu(
    train_vecs: np.ndarray,
    train_mds: Metadata,
    test_vecs: np.ndarray,
    test_mds: Metadata,
    train_cates: set,
    k=10,
    multi_tenant=True,
    num_procs=None,
):
    # Split test vectors and metadata into chunks
    num_procs = mp.cpu_count() if num_procs is None else num_procs
    chunk_size = math.ceil(len(test_vecs) / num_procs)
    chunks = [
        (test_vecs[i:i+chunk_size], test_mds[i:i+chunk_size], train_vecs, train_mds, train_cates, k, multi_tenant)
        for i in range(0, len(test_vecs), chunk_size)
    ]

    with mp.Pool(num_procs) as pool:
        results = pool.starmap(_compute_ground_truth_worker, chunks)

    ground_truth = [item for partial_res in results for item in partial_res]
    return np.array(ground_truth)


def compute_ground_truth_cuda(
    train_vecs: np.ndarray,
    train_mds: Metadata,
    test_vecs: np.ndarray,
    test_mds: Metadata,
    train_cates: set,
    k=10,
    multi_tenant=True,
    batch_size=1,
):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # here we store all the vectors in GPU memory
    train_vecs_pth = torch.from_numpy(train_vecs).unsqueeze(0).cuda()
    test_vecs_pth = torch.from_numpy(test_vecs).unsqueeze(1).cuda()

    # pre-compute all filter masks
    print("Pre-computing filter masks...")
    filter_mask_np = compute_filter_mask_parallel(train_mds, train_cates)
    filter_mask_pths = torch.tensor(filter_mask_np, device="cuda")
    filter_masks = {
        cate: filter_mask_pth
        for cate, filter_mask_pth in zip(sorted(train_cates), filter_mask_pths)
    }

    @torch.no_grad()
    def process_batch(test_vecs_batch, test_mds_batch):
        dists = torch.norm(train_vecs_pth - test_vecs_batch, dim=2)

        if multi_tenant:
            test_cates = [
                cate
                for test_md in test_mds_batch
                for cate in test_md
                if cate in train_cates
            ]

            if test_cates:
                n_repeat = torch.tensor(
                    [
                        sum(cate in test_cates for cate in test_md)
                        for test_md in test_mds_batch
                    ],
                    device="cuda",
                )

                combined_filter_mask = torch.stack(
                    [filter_masks[cate] for cate in test_cates], dim=0
                )

                repeated_dists = torch.repeat_interleave(dists, n_repeat, dim=0)
                combined_dists = torch.where(
                    combined_filter_mask, repeated_dists, float("inf")
                )

                top_k_indices = torch.argsort(combined_dists, dim=1)[:, :k]
                invalid_indices = combined_filter_mask.gather(1, top_k_indices) == 0
                top_k_indices.masked_fill_(invalid_indices, -1)
            else:
                top_k_indices = None
        else:
            top_k_indices = torch.argsort(dists, dim=1)[:, :k]

        return top_k_indices

    ground_truth = list()
    for i in tqdm(range(0, len(test_vecs_pth), batch_size)):
        test_vecs_batch = test_vecs_pth[i : i + batch_size]
        test_mds_batch = test_mds[i : i + batch_size]
        batch_results = process_batch(test_vecs_batch, test_mds_batch)
        if batch_results is not None:
            ground_truth.extend(batch_results.cpu().numpy())

    return np.array(ground_truth)


def load_sampled_metadata(
    train_mds: Metadata,
    test_mds: Metadata,
    train_cates: set[int],
    cache_dir: Path | None = None,
):
    """ Remove tenant IDs that are not in train_cates from train_mds and test_mds.
    """

    cache_dir = SAMPLED_METADATA_DIR if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    combined_bytes = (
        json.dumps(train_mds, sort_keys=True).encode()
        + json.dumps(test_mds, sort_keys=True).encode()
        + json.dumps(sorted([int(i) for i in train_cates]), sort_keys=True).encode()
    )
    combined_md5 = hashlib.md5(combined_bytes).hexdigest()

    pkl_path = cache_dir / f"{combined_md5}.pkl"
    if pkl_path.exists():
        print("Loading sampled metadata from cache %s..." % pkl_path)
        with open(pkl_path, "rb") as f:
            sampled_train_mds, sampled_test_mds = pkl.load(f)
    else:
        print("Sampling metadata...")
        sampled_train_mds = parallel_metadata_filter(
            train_mds, train_cates, mp.cpu_count()
        )
        sampled_test_mds = parallel_metadata_filter(
            test_mds, train_cates, mp.cpu_count()
        )

        with open(pkl_path, "wb") as f:
            pkl.dump((sampled_train_mds, sampled_test_mds), f)

    return sampled_train_mds, sampled_test_mds


""" Utility functions for parallel processing """


def _compute_set_union_worker(sets_chunk):
    result = set()
    for s in sets_chunk:
        result.update(s)
    return result


def parallel_set_union(all_sets, num_procs):
    # the length of the last chunk may be smaller than the others
    chunk_size = math.ceil(len(all_sets) / num_procs)
    chunks = [all_sets[i : i + chunk_size] for i in range(0, len(all_sets), chunk_size)]

    with mp.Pool(num_procs) as pool:
        partial_unions = pool.map(_compute_set_union_worker, chunks)

    return set.union(*partial_unions)


def _compute_ground_truth_worker(
    chunk_test_vecs, chunk_test_mds, train_vecs, train_mds, train_cates, k, multi_tenant
):
    partial_ground_truth = []

    for query_vec, test_md in zip(chunk_test_vecs, chunk_test_mds):
        dists = np.linalg.norm(train_vecs - query_vec, axis=1)

        if multi_tenant:
            for test_cate in test_md:
                if test_cate not in train_cates:
                    continue

                dists2 = dists.copy()
                filter_mask = np.array(
                    [test_cate in train_md for train_md in train_mds]
                )
                dists2[~filter_mask] = np.inf

                top_k_indices = np.argsort(dists2)[:k]
                if len(top_k_indices) < k:
                    top_k_indices = np.pad(
                        top_k_indices, (0, k - len(top_k_indices)), constant_values=-1
                    )
                partial_ground_truth.append(top_k_indices)

        else:
            top_k_indices = np.argsort(dists)[:k]
            if len(top_k_indices) < k:
                top_k_indices = np.pad(
                    top_k_indices, (0, k - len(top_k_indices)), constant_values=-1
                )
            partial_ground_truth.append(top_k_indices)

    return partial_ground_truth


def _compute_filter_mask_worker(train_mds_chunk, train_cates_sorted: np.ndarray):
    filter_mask = np.zeros(
        (len(train_cates_sorted), len(train_mds_chunk)), dtype=np.dtype("bool")
    )
    for i, train_md in enumerate(train_mds_chunk):
        cates = [cate for cate in train_md if cate in train_cates_sorted]
        indexes = np.searchsorted(train_cates_sorted, cates)
        filter_mask[indexes, i] = True
    return filter_mask


def compute_filter_mask_parallel(train_mds, train_cates, num_procs=mp.cpu_count()):
    dataset_size = len(train_mds)
    chunk_size = math.ceil(dataset_size / num_procs)
    train_mds_chunks = [
        train_mds[i : i + chunk_size] for i in range(0, dataset_size, chunk_size)
    ]

    train_cates_sorted = np.array(sorted(train_cates))
    with mp.Pool(num_procs) as pool:
        partial_filter_masks = pool.starmap(
            _compute_filter_mask_worker,
            [
                (train_mds_chunk, train_cates_sorted)
                for train_mds_chunk in train_mds_chunks
            ],
        )

    filter_mask = np.hstack(partial_filter_masks)
    return filter_mask


def _metadata_filter_worker(chunk, train_cates):
    return [[cate for cate in md if cate in train_cates] for md in chunk]


def parallel_metadata_filter(mds, train_cates, num_procs):
    # the length of the last chunk may be smaller than the others
    chunk_size = math.ceil(len(mds) / num_procs)
    chunks = [mds[i : i + chunk_size] for i in range(0, len(mds), chunk_size)]

    with mp.Pool(num_procs) as pool:
        results = pool.starmap(
            _metadata_filter_worker, [(chunk, train_cates) for chunk in chunks]
        )

    return [md for partial_res in results for md in partial_res]
