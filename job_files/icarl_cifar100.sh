#!/bin/bash
#SBATCH -p gpu_shared
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH -o /home/mayoughi/clvision-challenge-2022/output/baseline/icarl_cifar100/log.out
export PYTHONPATH=$PYTHONPATH:$HOME/clvision-challenge-2022/avalanche

module list
module load 2020
module load CUDA/11.0.2-GCC-9.3.0

cd clvision-challenge-2022/

python avalanche/examples/icarl.py \
