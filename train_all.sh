#!/bin/bash

mkdir -p logs

sbatch --job-name=sagittal_t1_left \
       --output=logs/sagittal_t1_left.out \
       --error=logs/sagittal_t1_left.err \
       --partition=ExternalResearch \
       -c 4 \
       --gres=gpu:v100:1 \
       --mem=16G \
       --wrap="python train_sagittal_t1_left_left.py"

sbatch --job-name=sagittal_t1_right \
       --output=logs/sagittal_t1_right.out \
       --error=logs/sagittal_t1_right.err \
       --partition=ExternalResearch \
       -c 4 \
       --gres=gpu:v100:1 \
       --mem=16G \
       --wrap="python train_sagittal_t1_right.py"

sbatch --job-name=sagittal_t2 \
       --output=logs/sagittal_t2.out \
       --error=logs/sagittal_t2.err \
       --partition=ExternalResearch \
       -c 4 \
       --gres=gpu:v100:1 \
       --mem=16G \
       --wrap="python train_sagittal_t2.py"

sbatch --job-name=axial_t2 \
       --output=logs/axial_t2.out \
       --error=logs/axial_t2.err \
       --partition=ExternalResearch \
       -c 4 \
       --gres=gpu:v100:1 \
       --mem=16G \
       --wrap="python train_axial_t2.py"
