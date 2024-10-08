#!/bin/bash

mkdir -p logs

source ~/.bashrc

conda activate RSNA

sbatch --job-name=sagittal_t1_left \
       --output=logs/sagittal_t1_left_%j.out \
       --error=logs/sagittal_t1_left_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_sagittal_t1_left.py"

sbatch --job-name=sagittal_t1_left \
       --output=logs/sagittal_t1_left_%j.out \
       --error=logs/sagittal_t1_left_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_sagittal_t1_left.py --all"

sbatch --job-name=sagittal_t1_right \
       --output=logs/sagittal_t1_right_%j.out \
       --error=logs/sagittal_t1_right_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_sagittal_t1_right.py"

sbatch --job-name=sagittal_t1_right \
       --output=logs/sagittal_t1_right_%j.out \
       --error=logs/sagittal_t1_right_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_sagittal_t1_right.py --all"

sbatch --job-name=sagittal_t2 \
       --output=logs/sagittal_t2_%j.out \
       --error=logs/sagittal_t2_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_sagittal_t2.py"

sbatch --job-name=sagittal_t2 \
       --output=logs/sagittal_t2_%j.out \
       --error=logs/sagittal_t2_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_sagittal_t2.py --all"

sbatch --job-name=axial_t2 \
       --output=logs/axial_t2_%j.out \
       --error=logs/axial_t2_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_axial_t2.py"

sbatch --job-name=axial_t2 \
       --output=logs/axial_t2_%j.out \
       --error=logs/axial_t2_%j.err \
       --partition=ExternalResearch \
       -c 16 \
       --gres=gpu:v100:1 \
       --mem=64G \
       --wrap="python train_axial_t2.py --all"
