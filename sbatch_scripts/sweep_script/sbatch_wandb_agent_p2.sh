#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pi_dagarwal_umass_edu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p2-sweep-agent
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100GB
#SBATCH --output=sbatch_scripts/slurm_output/p2-sweep-agent-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p2-sweep-agent-%j.err

set -euo pipefail

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion

cd /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory

if [[ -z "${SWEEP_ID:-}" ]]; then
  echo "ERROR: SWEEP_ID is required."
  echo "Set SWEEP_ID to either <entity>/<project>/<sweep_id> or just <sweep_id> with WANDB_ENTITY and WANDB_PROJECT set."
  exit 1
fi

if [[ "${SWEEP_ID}" == */*/* ]]; then
  SWEEP_TARGET="${SWEEP_ID}"
else
  if [[ -z "${WANDB_ENTITY:-}" || -z "${WANDB_PROJECT:-}" ]]; then
    echo "ERROR: WANDB_ENTITY and WANDB_PROJECT are required when SWEEP_ID is not fully qualified."
    exit 1
  fi
  SWEEP_TARGET="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
fi

AGENT_COUNT="${WANDB_AGENT_COUNT:-1}"

nvidia-smi
echo "Starting W&B agent"
echo "  Host: $(hostname)"
echo "  Sweep target: ${SWEEP_TARGET}"
echo "  Agent count: ${AGENT_COUNT}"

wandb agent --count "${AGENT_COUNT}" "${SWEEP_TARGET}"
