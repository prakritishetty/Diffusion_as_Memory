#!/bin/bash
set -euo pipefail

# Submit multiple SLURM jobs, each running a W&B agent.
#
# Example:
#   bash sbatch_scripts/launch_wandb_sweep_agents.sh \
#     --sweep-id qbfan4dk \
#     --project diffusion-as-memory \
#     --num-agents 4 \
#     --agent-count-per-job 2 \
#     --entity balachandradevarangadi-umass-amherst


SWEEP_ID=""
WANDB_ENTITY=""
WANDB_PROJECT=""
NUM_AGENTS=1
AGENT_COUNT_PER_JOB=1
JOB_PREFIX="p2-sweep"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep-id)
      SWEEP_ID="$2"
      shift 2
      ;;
    --entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --num-agents)
      NUM_AGENTS="$2"
      shift 2
      ;;
    --agent-count-per-job)
      AGENT_COUNT_PER_JOB="$2"
      shift 2
      ;;
    --job-prefix)
      JOB_PREFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: --sweep-id is required."
  exit 1
fi

if [[ ! "${SWEEP_ID}" == */*/* ]]; then
  if [[ -z "${WANDB_ENTITY}" || -z "${WANDB_PROJECT}" ]]; then
    echo "ERROR: provide --entity and --project when --sweep-id is not fully qualified."
    exit 1
  fi
fi

mkdir -p sbatch_scripts/slurm_output

echo "Submitting ${NUM_AGENTS} W&B agent jobs..."
for i in $(seq 1 "${NUM_AGENTS}"); do
  job_name="${JOB_PREFIX}-${i}"
  sbatch \
    --job-name "${job_name}" \
    --export=ALL,SWEEP_ID="${SWEEP_ID}",WANDB_ENTITY="${WANDB_ENTITY}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_AGENT_COUNT="${AGENT_COUNT_PER_JOB}" \
    sbatch_scripts/sweep_script/sbatch_wandb_agent_p2.sh
  echo "  submitted ${job_name}"
done

