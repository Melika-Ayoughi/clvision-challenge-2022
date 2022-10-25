#!/bin/bash

#SBATCH -p gpu_titanrtx
#SBATCH --gpus=1
#SBATCH --constraint=gold_5118
#SBATCH -t 00:30:00
#SBATCH -o /home/mayoughi/clvision-challenge-2022/output/baseline/icarl_ego/log.out
export PYTHONPATH=$PYTHONPATH:$HOME/clvision-challenge-2022/avalanche

module list
module load 2020
module load CUDA/11.0.2-GCC-9.3.0

cd clvision-challenge-2022/

python starting_template_instance_classification.py \
--EXP_NAME "icarl_ego" \
--baseline "icarl"
#python run.py
