#!/usr/bin/env bash
# Full retrieval and generation evaluation for the tax RAG pipeline.
#
# Submit from the project root:
#   sbatch slurm/run_rag_eval.sh
#
# The generated SLURM .out log contains expected answers, retrieved chunks,
# generated answers, and retrieval hit summaries.

#SBATCH --job-name=tax-rag-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
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

RESULTS_JSONL="logs/tax-rag-eval-${SLURM_JOB_ID:-manual}.jsonl"

echo "Running evaluation in project root: $PROJECT_ROOT"
echo "Using container: $CONTAINER"
echo "Writing structured results to: $RESULTS_JSONL"

singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.rag_cli \
  --build-index \
  --raw-dir data/raw

singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.evaluate_rag \
  --questions evaluation/tax_eval_questions.jsonl \
  --results-jsonl "$RESULTS_JSONL" \
  --top-k 3 \
  --max-new-tokens 512
