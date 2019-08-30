#!/bin/bash
#SBATCH --job-name=mlp2nb
#SBATCH --output=./slurm_out/mlp2nb_%A_%a.out
#SBATCH --error=./slurm_out/mlp2nb_%A_%a.err
#SBATCH --workdir=/clusterFS/home/student/deekay/BinaryNet/Train-time
#SBATCH --export=HOME=/clusterFS/home/student/deekay,PATH
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,gpu2,gpu6
#SBATCH --array=0-50:10%4
#SBATCH --gres=gpu:TeslaK40c:1
#SBATCH --mem 6G
#SBATCH --cpus-per-gpu=2

source $HOME/.bashrc
conda activate theano
python mnist.py --binary -a ${SLURM_ARRAY_TASK_ID}

