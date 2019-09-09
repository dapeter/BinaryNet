#!/bin/bash
#SBATCH --job-name=nconv4b
#SBATCH --output=./slurm_out/nconv4b_%A_%a.out
#SBATCH --error=./slurm_out/nconv4b_%A_%a.err
#SBATCH --workdir=/clusterFS/home/student/deekay/BinaryNet/Train-time
#SBATCH --export=HOME=/clusterFS/home/student/deekay,PATH
#SBATCH --partition=gpu,gpu2,gpu6
#SBATCH --array=1-10:2%2
#SBATCH --gres=gpu:TeslaK40c:1
#SBATCH --mem=16

source $HOME/.bashrc
conda activate theano
python cifar10_big.py --binary -a ${SLURM_ARRAY_TASK_ID}

