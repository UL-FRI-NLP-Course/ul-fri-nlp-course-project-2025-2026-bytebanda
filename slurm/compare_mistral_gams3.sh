#!/usr/bin/env bash
# Compare Mistral and GaMS3-12B as answer generators on the same RAG retrieval.
#
# Submit from the project root:
#   CONTAINER=/path/to/container.sif sbatch slurm/compare_mistral_gams3.sh
#
# Optional overrides:
#   LIMIT=10 QUESTIONS=evaluation/tax_eval_questions.jsonl sbatch ...

#SBATCH --job-name=rag-model-compare
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch, not run directly."
  echo "Use: LIMIT=5 sbatch slurm/compare_mistral_gams3.sh"
  exit 1
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER="${CONTAINER:-/d/hpc/projects/onj_fri/trije_konjeniki_apokalipse/containers/container-pytorch-rag.sif}"
RUN_ID="${SLURM_JOB_ID:-manual}"

QUESTIONS="${QUESTIONS:-evaluation/tax_eval_questions.jsonl}"
LIMIT="${LIMIT:-5}"
TOP_K="${TOP_K:-3}"
CANDIDATE_K="${CANDIDATE_K:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-prompts/tax_assistant_strict_prompt.txt}"

MISTRAL_MODEL="${MISTRAL_MODEL:-/d/hpc/projects/onj_fri/models/intent}"
GAMS3_MODEL="${GAMS3_MODEL:-/d/hpc/projects/onj_fri/dreitokens/models/GaMS3-12B-Instruct}"

INDEX_DIR="${INDEX_DIR:-data/eval/model-compare-legal}"
MISTRAL_RESULTS="logs/model-compare-mistral-${RUN_ID}.jsonl"
GAMS3_RESULTS="logs/model-compare-gams3-12b-${RUN_ID}.jsonl"

cd "$PROJECT_ROOT"
mkdir -p logs "$INDEX_DIR"

export PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Project root: $PROJECT_ROOT"
echo "Container: $CONTAINER"
echo "Questions: $QUESTIONS"
echo "Limit: $LIMIT"
echo "Mistral model: $MISTRAL_MODEL"
echo "GaMS3 model: $GAMS3_MODEL"
echo
echo "=== GPU ==="
nvidia-smi || true

echo
echo "=== Building shared legal index ==="
singularity exec --nv "$CONTAINER" python -m src.rag_cli \
  --build-index \
  --raw-dir downloads/pisrs \
  --chunk-strategy legal \
  --chunk-size 1800 \
  --overlap 150 \
  --processed-chunks-path "$INDEX_DIR/processed_chunks.jsonl" \
  --index-path "$INDEX_DIR/faiss.index" \
  --index-chunks-path "$INDEX_DIR/chunks.jsonl"

echo
echo "=== Evaluating Mistral generator ==="
singularity exec --nv "$CONTAINER" python -m src.evaluate_rag \
  --run-label mistral-generator \
  --prompt-label strict \
  --questions "$QUESTIONS" \
  --results-jsonl "$MISTRAL_RESULTS" \
  --index-path "$INDEX_DIR/faiss.index" \
  --chunks-path "$INDEX_DIR/chunks.jsonl" \
  --model-path "$MISTRAL_MODEL" \
  --system-prompt "$SYSTEM_PROMPT" \
  --retrieval-mode hybrid \
  --candidate-k "$CANDIDATE_K" \
  --top-k "$TOP_K" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT"

echo
echo "=== Evaluating GaMS3-12B generator ==="
singularity exec --nv "$CONTAINER" python -m src.evaluate_rag \
  --run-label gams3-12b-generator \
  --prompt-label strict \
  --questions "$QUESTIONS" \
  --results-jsonl "$GAMS3_RESULTS" \
  --index-path "$INDEX_DIR/faiss.index" \
  --chunks-path "$INDEX_DIR/chunks.jsonl" \
  --model-path "$GAMS3_MODEL" \
  --system-prompt "$SYSTEM_PROMPT" \
  --retrieval-mode hybrid \
  --candidate-k "$CANDIDATE_K" \
  --top-k "$TOP_K" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT"

echo
echo "=== Comparison ==="
singularity exec "$CONTAINER" python -m src.compare_eval_results \
  "mistral=$MISTRAL_RESULTS" \
  "gams3_12b=$GAMS3_RESULTS"

echo
echo "Result files:"
echo "- $MISTRAL_RESULTS"
echo "- $GAMS3_RESULTS"
