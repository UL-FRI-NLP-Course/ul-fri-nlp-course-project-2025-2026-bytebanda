#!/usr/bin/env bash
# Minimal SLURM smoke test for the tax RAG pipeline.
#
# Submit from the project root:
#   sbatch slurm/run_rag_test.sh
#
#SBATCH --job-name=tax-rag-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/tax_project}"
CONTAINER="${CONTAINER:-$HOME/containers/pytorch.sif}"
USE_GPU="${USE_GPU:-1}"

cd "$PROJECT_ROOT"
mkdir -p logs data/processed data/index

SINGULARITY_ARGS=()
if [[ "$USE_GPU" == "1" ]]; then
  SINGULARITY_ARGS+=(--nv)
fi

echo "Running in project root: $PROJECT_ROOT"
echo "Using container: $CONTAINER"

singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.rag_cli --build-index --raw-dir downloads/pisrs
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.rag_cli --ask "Kaj je DDV?"
