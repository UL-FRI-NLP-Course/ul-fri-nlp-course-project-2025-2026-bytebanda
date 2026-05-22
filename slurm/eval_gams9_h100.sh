#!/usr/bin/env bash
# Evaluate GaMS-9B-Instruct as the RAG answer generator on an H100 GPU.
#
# Submit from the project root:
#   LIMIT=5 sbatch slurm/eval_gams9_h100.sh
#
# Optional comparison against an existing Mistral result:
#   MISTRAL_RESULTS=logs/model-compare-mistral-15713657.jsonl sbatch ...

#SBATCH --job-name=rag-gams9-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch, not run directly."
  echo "Use: LIMIT=5 sbatch slurm/eval_gams9_h100.sh"
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

GAMS9_MODEL="${GAMS9_MODEL:-/d/hpc/projects/onj_fri/brainstorm/models/GaMS-9B-Instruct}"
INDEX_DIR="${INDEX_DIR:-data/eval/model-compare-legal}"
GAMS9_RESULTS="logs/model-compare-gams9-${RUN_ID}.jsonl"
MISTRAL_RESULTS="${MISTRAL_RESULTS:-}"

cd "$PROJECT_ROOT"
mkdir -p logs "$INDEX_DIR"

export PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Project root: $PROJECT_ROOT"
echo "Container: $CONTAINER"
echo "Questions: $QUESTIONS"
echo "Limit: $LIMIT"
echo "GaMS-9B model: $GAMS9_MODEL"
echo "Index dir: $INDEX_DIR"
echo
echo "=== GPU ==="
nvidia-smi || true

if [[ ! -s "$INDEX_DIR/faiss.index" || ! -s "$INDEX_DIR/chunks.jsonl" ]]; then
  echo
  echo "=== Building shared legal index ==="
  singularity exec --nv "$CONTAINER" python -m src.rag_cli \
    --build-index \
    --raw-dir data/raw \
    --chunk-strategy legal \
    --chunk-size 1800 \
    --overlap 150 \
    --processed-chunks-path "$INDEX_DIR/processed_chunks.jsonl" \
    --index-path "$INDEX_DIR/faiss.index" \
    --index-chunks-path "$INDEX_DIR/chunks.jsonl"
else
  echo
  echo "=== Reusing existing index ==="
  echo "$INDEX_DIR/faiss.index"
  echo "$INDEX_DIR/chunks.jsonl"
fi

echo
echo "=== Evaluating GaMS-9B generator ==="
singularity exec --nv "$CONTAINER" python -m src.evaluate_rag \
  --run-label gams9-generator \
  --prompt-label strict \
  --questions "$QUESTIONS" \
  --results-jsonl "$GAMS9_RESULTS" \
  --index-path "$INDEX_DIR/faiss.index" \
  --chunks-path "$INDEX_DIR/chunks.jsonl" \
  --model-path "$GAMS9_MODEL" \
  --system-prompt "$SYSTEM_PROMPT" \
  --retrieval-mode hybrid \
  --candidate-k "$CANDIDATE_K" \
  --top-k "$TOP_K" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT"

if [[ -n "$MISTRAL_RESULTS" && -s "$MISTRAL_RESULTS" ]]; then
  echo
  echo "=== Comparison ==="
  singularity exec "$CONTAINER" python -m src.compare_eval_results \
    "mistral=$MISTRAL_RESULTS" \
    "gams9=$GAMS9_RESULTS"
fi

echo
echo "Result file:"
echo "- $GAMS9_RESULTS"
