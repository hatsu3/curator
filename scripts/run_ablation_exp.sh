log_dir=logs_ablation
python=/home/xxx/miniconda3/envs/ann_bench/bin/python

mkdir -p ${log_dir}

# yfcc100m

# shared_ivf on yfcc100m
sudo ${python} \
run_parallel_exp.py run_shared_ivf_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 0 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_ivf_on_yfcc100m.log

# ivf_bf on yfcc100m
sudo ${python} \
run_parallel_exp.py run_shared_ivf_bf_overall_exp \
  --dataset yfcc100m \
  --gamma 96 \
  --cpu-limit 1 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_ivf_bf_on_yfcc100m.log

# curator_flat on yfcc100m
sudo ${python} \
run_parallel_exp.py run_curator_flat_search_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 2 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/curator_flat_on_yfcc100m.log

# curator on yfcc100m
sudo ${python} \
run_parallel_exp.py run_curator_overall_exp \
  --dataset yfcc100m \
  --cpu-limit 3 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/curator_on_yfcc100m.log

# arxiv-large

# shared_ivf on arxiv-large
sudo ${python} \
run_parallel_exp.py run_shared_ivf_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 4 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_ivf_on_arxiv_large.log

# ivf_bf on arxiv-large
sudo ${python} \
run_parallel_exp.py run_shared_ivf_bf_overall_exp \
  --dataset arxiv-large \
  --gamma 768 \
  --cpu-limit 5 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/shared_ivf_bf_on_arxiv_large.log

# curator_flat on arxiv-large
sudo ${python} \
run_parallel_exp.py run_curator_flat_search_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 6 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/curator_flat_on_arxiv_large.log

# curator on arxiv-large
sudo ${python} \
run_parallel_exp.py run_curator_overall_exp \
  --dataset arxiv-large \
  --cpu-limit 7 \
  --mem_limit 20000000000 \
  --num_runs 1 \
|& tee ${log_dir}/curator_on_arxiv_large.log
