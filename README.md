# Curator: Efficient Indexing for Multi-Tenant Vector Databases

Curator is an in-memory vector index tailored for multi-tenant queries that simultaneously achieves low memory overhead and high query performance. Curator indexes each tenant’s vectors with a tenant-specific clustering tree and encodes these trees compactly as sub-trees of a shared clustering tree. Each tenant’s clustering tree dynamically adapts to its unique vector distribution, while maintaining a low per-tenant memory footprint. 

Please refer to our paper for more details: [Curator: Efficient Indexing for Multi-Tenant Vector Databases](https://arxiv.org/abs/2401.07119).

## Repository Structure

- `3rd_party/faiss`: C++ impl of Curator and baselines
  
  - `MultiTenantIndexIVFHierarchical.cpp`: Curator
  - `MultiTenantIndexIVFFlat.cpp`: IVF with metadata filtering
  - `MultiTenantIndexIVFFlatSep.cpp`: IVF with per-tenant indexing
  - `MultiTenantIndexHNSW.cpp`: HNSW with metadata filtering

- `indexes`: Python API for indexes

  - `ivf_hier_faiss.py`: Curator
  - `ivf_flat_mt_faiss.py`: IVF with metadata filtering
  - `ivf_flat_sepidx_faiss.py`: IVF with per-tenant indexing
  - `hnsw_mt_hnswlib.py`: HNSW with metadata filtering
  - `hnsw_sepidx_hnswlib.py`: HNSW with per-tenant indexing

- `dataset`: code for evaluation datasets

  - `arxiv_dataset.py`: arXiv dataset
  - `yfcc100m_dataset.py`: YFCC100M dataset
  - `randperm_dataset.py`: synthetic dataset with randomized access metadata (used in ablation study)

- `benchmark`: code for running benchmarks

- `scripts`: scripts for running experiments

  - `run_ablation.sh`: run ablation study experiments
  - `run_mt_search.sh`: run multi-threaded search experiments
  - `run_overall_exp.sh`: evaluate overall performance of indexes (query, insert, delete, index size)
  - `run_param_sweep.sh`: run parameter sweep experiments

- `plotting`: results of experiments and scripts for plotting

## How to Use

### Install Dependencies

We assume that you have installed Anaconda. To install the required Python packages, run the following command:

```bash
conda env create -f environment.yml -n ann_bench
conda activate ann_bench
```

### Build from Source

```bash
cd 3rd_party/faiss

cmake -B build . \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBUILD_TESTING=ON

make -C build -j32 faiss_avx2
make -C build -j32 swigfaiss_avx2
cd build/faiss/python
python setup.py install
```

### Generate Datasets

```bash
mkdir -p data/yfcc100m
yfcc100m_base_url="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M"
wget -P data/yfcc100m ${yfcc100m_base_url}/base.10M.u8bin
wget -P data/yfcc100m ${yfcc100m_base_url}/base.metadata.10M.spmat

mkdir -p data/arxiv
# manually download arxiv dataset from https://www.kaggle.com/datasets/Cornell-University/arxiv
# and put it at data/arxiv/arxiv-metadata-oai-snapshot.json

python -m dataset.yfcc100m_dataset
python -m dataset.arxiv_dataset
```

### Build Docker Image

```bash
# Download the cuda-keyring package for updating the CUDA linux GPG repository key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# Please replace $distro and $arch with your own distro and arch
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
sudo docker build --rm -t ann-bench .
```

### Run Benchmarks

Please refer to scripts in `scripts` folder for details. For example, to evaluate Curator on YFCC100M dataset, run the following command:

```bash
python=$(which python)  # assuming conda env is activated

sudo ${python} \
run_parallel_exp.py run_curator_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 0 \
  --mem_limit 20000000000 \
  --num_runs 1
```
