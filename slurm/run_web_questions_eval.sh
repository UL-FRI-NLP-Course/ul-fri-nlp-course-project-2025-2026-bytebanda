#!/usr/bin/env bash
# GPU evaluation for evaluation/generated_web_questions.json.
#
# Submit from the project root:
#   sbatch slurm/run_web_questions_eval.sh
#
# For a short smoke test:
#   LIMIT=2 sbatch slurm/run_web_questions_eval.sh

#SBATCH --job-name=web-rag-eval
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
LIMIT="${LIMIT:-0}"
RUN_ID="${SLURM_JOB_ID:-manual}"

cd "$PROJECT_ROOT"
mkdir -p logs data/processed data/index

SINGULARITY_ARGS=()
if [[ "$USE_GPU" == "1" ]]; then
  SINGULARITY_ARGS+=(--nv)
fi

RESULTS_JSONL="logs/web-question-eval-${RUN_ID}.jsonl"

echo "Running web-question evaluation in project root: $PROJECT_ROOT"
echo "Using container: $CONTAINER"
echo "Writing structured results to: $RESULTS_JSONL"
echo "Checking GPU visibility"

singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" nvidia-smi

COMMAND=(
  python -m src.evaluate_web_questions
  --questions evaluation/generated_web_questions.json
  --results-jsonl "$RESULTS_JSONL"
  --retrieval-mode hybrid
  --candidate-k 100
  --top-k 3
  --max-new-tokens 512
  --gpu-smoke-test
)

if [[ "$LIMIT" != "0" ]]; then
  COMMAND+=(--limit "$LIMIT")
fi

singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" "${COMMAND[@]}"
