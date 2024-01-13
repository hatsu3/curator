log_dir=logs_param_sweep
python=/home/xxx/miniconda3/envs/ann_bench/bin/python

mkdir -p ${log_dir}

# index = curator, shared_ivf, separate_ivf, shared_hnsw, separate_hnsw
index=${1:-curator}
cpu_limit=${2:-0}
param1=$3
param2=$4
param3=$5
param4=$6

if [ ${index} == "curator" ]; then
    sudo ${python} \
    run_parallel_exp.py run_${index}_param_sweep \
        --nlist_space "${param1}" \
        --gamma1_space "${param2}" \
        --gamma2_space "${param3}" \
        --max_sl_size_space "${param4}" \
        --log_dir ${log_dir}/${index} \
        --cpu_limit ${cpu_limit} \
        --mem_limit 20000000000

elif [ ${index} == "shared_ivf" ]; then
    sudo ${python} \
    run_parallel_exp.py run_${index}_param_sweep \
        --nlist_space "${param1}" \
        --nprobe_space "${param2}" \
        --log_dir ${log_dir}/${index} \
        --cpu_limit ${cpu_limit} \
        --mem_limit 20000000000

elif [ ${index} == "separate_ivf" ]; then
    sudo ${python} \
    run_parallel_exp.py run_${index}_param_sweep \
        --nlist_space "${param1}" \
        --nprobe_space "${param2}" \
        --log_dir ${log_dir}/${index} \
        --cpu_limit ${cpu_limit} \
        --mem_limit 100000000000

elif [ ${index} == "shared_hnsw" ]; then
    sudo ${python} \
    run_parallel_exp.py run_${index}_param_sweep \
        --construction_ef_space "${param1}" \
        --search_ef_space "${param2}" \
        --m_space "${param3}" \
        --log_dir ${log_dir}/${index} \
        --cpu_limit ${cpu_limit} \
        --mem_limit 20000000000

elif [ ${index} == "separate_hnsw" ]; then
    sudo ${python} \
    run_parallel_exp.py run_${index}_param_sweep \
        --construction_ef_space "${param1}" \
        --search_ef_space "${param2}" \
        --m_space "${param3}" \
        --log_dir ${log_dir}/${index} \
        --cpu_limit ${cpu_limit} \
        --mem_limit 100000000000
fi
