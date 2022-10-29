#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J gocattranse
#SBATCH -o out/gocattranse.%J.out
#SBATCH -e err/gocattranse.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=100G
#SBATCH --constraint=[cascadelake]


function readJobArrayParams () {
    epochs_f=${1}
    emb_size=${2}
    epochs_s=${3}
    lr=${4}
    margin=${5}
}

function getJobArrayParams () {
  local job_params_file="params_transe_only_train.txt"

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
python run_cge_transe.py -case go -g categorical -epf ${epochs_f} -esize ${emb_size} -eps ${epochs_s} -lr ${lr} -m ${margin} -dev cpu -train

