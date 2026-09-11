#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=unieval
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40GB
#SBATCH --output=slurm_output/unieval-%j.out
#SBATCH --error=slurm_output/unieval-%j.err

echo "running unieval factual consistency"

module load conda/latest
conda activate ./myvenv     

python ../evaluation/run_uni_eval_factutal_consistency.py

echo "unieval complete!"