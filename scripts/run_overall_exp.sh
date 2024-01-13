log_dir=logs_overall
python=/home/xxx/miniconda3/envs/ann_bench/bin/python

mkdir -p ${log_dir}

# yfcc100m

# curator on yfcc100m
sudo ${python} \
run_parallel_exp.py run_curator_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 0 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/curator_on_yfcc100m.log

# shared_ivf on yfcc100m
sudo ${python} \
run_parallel_exp.py run_shared_ivf_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 1 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_ivf_on_yfcc100m.log

# separate_ivf on yfcc100m
sudo ${python} \
run_parallel_exp.py run_separate_ivf_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 2 \
  --mem_limit 60000000000 \
  --num_runs 1 \
|& tee ${log_dir}/separate_ivf_on_yfcc100m.log

# shared_hnsw on yfcc100m
sudo ${python} \
run_parallel_exp.py run_shared_hnsw_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 3 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_hnsw_on_yfcc100m.log

# separate_hnsw on yfcc100m
sudo ${python} \
run_parallel_exp.py run_separate_hnsw_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 4 \
  --mem_limit 60000000000 \
  --num_runs 1 \
|& tee ${log_dir}/separate_hnsw_on_yfcc100m.log


# arxiv-large

# curator on arxiv-large
sudo ${python} \
run_parallel_exp.py run_curator_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 5 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/curator_on_arxiv_large.log

# shared_ivf on arxiv-large
sudo ${python} \
run_parallel_exp.py run_shared_ivf_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 6 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_ivf_on_arxiv_large.log

# separate_ivf on arxiv-large
sudo ${python} \
run_parallel_exp.py run_separate_ivf_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 7 \
  --mem_limit 100000000000 \
  --num_runs 1 \
|& tee ${log_dir}/separate_ivf_on_arxiv_large.log

# shared_hnsw on arxiv-large
sudo ${python} \
run_parallel_exp.py run_shared_hnsw_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 8 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_hnsw_on_arxiv_large.log

# separate_hnsw on arxiv-large
sudo ${python} \
run_parallel_exp.py run_separate_hnsw_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 9 \
  --mem_limit 100000000000 \
  --num_runs 1 \
|& tee ${log_dir}/separate_hnsw_on_arxiv_large.log
