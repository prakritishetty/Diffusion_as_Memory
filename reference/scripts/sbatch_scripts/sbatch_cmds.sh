#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pi_dagarwal_umass_edu
#SBATCH --gpus=l40s:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p0-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB

#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=denoiser-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a100"
#SBATCH --nodes=1