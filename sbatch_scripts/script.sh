#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=0-08:00:00
#SBATCH --job-name=sbatch_testing
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

python testing.py