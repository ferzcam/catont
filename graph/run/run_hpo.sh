#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J go
#SBATCH -o out/go.%J.out
#SBATCH -e err/go.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=100G
#SBATCH --constraint=[cascadelake]


function readJobArrayParams () {
    num_walks=${1}
    walk_length=${2}
    alpha=${3}
    epochs_w2v=${4}
    window_size=${5}
    embedding_size=${6}
    epochs_mlp=${7}
    learning_rate=${8}
}

function getJobArrayParams () {
  local job_params_file="params_random_walk.txt"

  if [ -z "${SLURM_ARRAY_TASK_ID}" ] ; then
    echo "ERROR: Require job array.  Use '--array=#-#', or '--array=#,#,#' to specify."
    echo " Array #'s are 1-based, and correspond to line numbers in job params file: ${job_params_file}"
    exit 1
  fi

  if [ ! -f "$job_params_file" ] ; then  
    echo "Missing job parameters file ${job_params_file}"
    exit 1
  fi

  readJobArrayParams $(head ${job_params_file} -n ${SLURM_ARRAY_TASK_ID} | tail -n 1)
}

getJobArrayParams

# Run the code
python run_cge_random_walk.py -case go -g owl2vecstar -nwalks ${num_walks} -wlen ${walk_length} -alpha ${alpha} -epw2v ${epochs_w2v} -wsize ${window_size} -esize ${embedding_size} -epmlp ${epochs_mlp} -lr ${learning_rate} -dev cuda -train -test

