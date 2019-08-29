#!/bin/bash
#SBATCH --job-name=nn
#SBATCH --workdir=/clusterFS/home/student/deekay/BinaryNet/Train-time
#SBATCH --output=./slurm_out/nn_%A_%a.out
#SBATCH --error=./slurm_out/nn_%A_%a.err
#SBATCH --export=HOME=/clusterFS/home/student/deekay,PATH
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,gpu2,gpu6
#SBATCH --gres=gpu:GeForceGTX1080Ti:1
#SBATCH --mem 4G

source $HOME/.bashrc
conda activate theano
python mnist.py --binary -a 20

