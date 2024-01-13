log_dir=logs_mt_search
python=/home/xxx/miniconda3/envs/ann_bench/bin/python

mkdir -p ${log_dir}

# index = curator, shared_ivf, separate_ivf, shared_hnsw, separate_hnsw
# parallel_mode = intra, inter
# num_threads = 1, 2, 4, 8, 16
# mem_limit = 20000000000, 100000000000 (20GB, 100GB)

index=${1:-curator}
parallel_mode=${2:-intra}
num_threads=${3:-1}
cpu_limit=${4:-0}
mem_limit=${5:-20g}

if [ ${mem_limit} == "20g" ]; then
  mem_limit=20000000000
elif [ ${mem_limit} == "100g" ]; then
  mem_limit=100000000000
fi

sudo ${python} \
run_parallel_exp.py run_${index}_mt_exp \
  --num_threads ${num_threads} \
  --parallel_mode ${parallel_mode} \
  --cpu_limit ${cpu_limit} \
  --mem_limit ${mem_limit} \
|& tee ${log_dir}/${index}_${parallel_mode}_t${num_threads}_on_yfcc100m.log
