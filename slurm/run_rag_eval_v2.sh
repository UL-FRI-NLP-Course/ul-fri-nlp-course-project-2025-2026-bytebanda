#!/usr/bin/env bash
# Side-by-side evaluation for the improved tax RAG pipeline.
#
# Submit from the project root:
#   sbatch slurm/run_rag_eval_v2.sh
#
# It builds:
#   1. baseline fixed chunks + dense retrieval
#   2. improved legal-article chunks + hybrid reranking
#
# It then compares retrieval-only results and three generation prompts.

#SBATCH --job-name=tax-rag-eval-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/tax_project}"
CONTAINER="${CONTAINER:-$HOME/containers/pytorch.sif}"
USE_GPU="${USE_GPU:-1}"
RUN_ID="${SLURM_JOB_ID:-manual}"

cd "$PROJECT_ROOT"
mkdir -p logs data/eval/baseline-fixed data/eval/legal-article

export PYTHONUNBUFFERED=1

SINGULARITY_ARGS=()
if [[ "$USE_GPU" == "1" ]]; then
  SINGULARITY_ARGS+=(--nv)
fi

BASELINE_DIR="data/eval/baseline-fixed"
LEGAL_DIR="data/eval/legal-article"

BASELINE_RETRIEVAL="logs/rag-v2-baseline-fixed-dense-${RUN_ID}.jsonl"
LEGAL_RETRIEVAL="logs/rag-v2-legal-hybrid-retrieval-${RUN_ID}.jsonl"
LEGAL_PROMPT_BASELINE="logs/rag-v2-legal-hybrid-prompt-baseline-${RUN_ID}.jsonl"
LEGAL_PROMPT_STRICT="logs/rag-v2-legal-hybrid-prompt-strict-${RUN_ID}.jsonl"
LEGAL_PROMPT_EXTRACTIVE="logs/rag-v2-legal-hybrid-prompt-extractive-${RUN_ID}.jsonl"

echo "Running RAG v2 evaluation in project root: $PROJECT_ROOT"
echo "Using container: $CONTAINER"
echo "RUN_ID=$RUN_ID"

echo
echo "=== Building baseline fixed-chunk index ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.rag_cli \
  --build-index \
  --raw-dir downloads/pisrs \
  --chunk-strategy fixed \
  --chunk-size 1200 \
  --overlap 200 \
  --processed-chunks-path "$BASELINE_DIR/processed_chunks.jsonl" \
  --index-path "$BASELINE_DIR/faiss.index" \
  --index-chunks-path "$BASELINE_DIR/chunks.jsonl"

echo
echo "=== Evaluating baseline dense retrieval ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.evaluate_rag \
  --run-label baseline-fixed-dense \
  --prompt-label none \
  --questions evaluation/tax_eval_questions.jsonl \
  --index-path "$BASELINE_DIR/faiss.index" \
  --chunks-path "$BASELINE_DIR/chunks.jsonl" \
  --retrieval-mode dense \
  --top-k 3 \
  --no-generate \
  --results-jsonl "$BASELINE_RETRIEVAL"

echo
echo "=== Building improved legal-article index ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.rag_cli \
  --build-index \
  --raw-dir downloads/pisrs \
  --chunk-strategy legal \
  --chunk-size 1800 \
  --overlap 150 \
  --processed-chunks-path "$LEGAL_DIR/processed_chunks.jsonl" \
  --index-path "$LEGAL_DIR/faiss.index" \
  --index-chunks-path "$LEGAL_DIR/chunks.jsonl"

echo
echo "=== Evaluating improved hybrid retrieval ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.evaluate_rag \
  --run-label legal-hybrid-retrieval \
  --prompt-label none \
  --questions evaluation/tax_eval_questions.jsonl \
  --index-path "$LEGAL_DIR/faiss.index" \
  --chunks-path "$LEGAL_DIR/chunks.jsonl" \
  --retrieval-mode hybrid \
  --candidate-k 100 \
  --lexical-weight 0.25 \
  --source-boost 0.20 \
  --top-k 3 \
  --no-generate \
  --results-jsonl "$LEGAL_RETRIEVAL"

echo
echo "=== Evaluating improved hybrid generation with baseline prompt ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.evaluate_rag \
  --run-label legal-hybrid-prompt-baseline \
  --prompt-label baseline \
  --questions evaluation/tax_eval_questions.jsonl \
  --index-path "$LEGAL_DIR/faiss.index" \
  --chunks-path "$LEGAL_DIR/chunks.jsonl" \
  --retrieval-mode hybrid \
  --candidate-k 100 \
  --lexical-weight 0.25 \
  --source-boost 0.20 \
  --top-k 3 \
  --system-prompt prompts/tax_assistant_system_prompt.txt \
  --max-new-tokens 768 \
  --results-jsonl "$LEGAL_PROMPT_BASELINE"

echo
echo "=== Evaluating improved hybrid generation with strict prompt ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.evaluate_rag \
  --run-label legal-hybrid-prompt-strict \
  --prompt-label strict \
  --questions evaluation/tax_eval_questions.jsonl \
  --index-path "$LEGAL_DIR/faiss.index" \
  --chunks-path "$LEGAL_DIR/chunks.jsonl" \
  --retrieval-mode hybrid \
  --candidate-k 100 \
  --lexical-weight 0.25 \
  --source-boost 0.20 \
  --top-k 3 \
  --system-prompt prompts/tax_assistant_strict_prompt.txt \
  --max-new-tokens 768 \
  --results-jsonl "$LEGAL_PROMPT_STRICT"

echo
echo "=== Evaluating improved hybrid generation with extractive prompt ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.evaluate_rag \
  --run-label legal-hybrid-prompt-extractive \
  --prompt-label extractive \
  --questions evaluation/tax_eval_questions.jsonl \
  --index-path "$LEGAL_DIR/faiss.index" \
  --chunks-path "$LEGAL_DIR/chunks.jsonl" \
  --retrieval-mode hybrid \
  --candidate-k 100 \
  --lexical-weight 0.25 \
  --source-boost 0.20 \
  --top-k 3 \
  --system-prompt prompts/tax_assistant_extractive_prompt.txt \
  --max-new-tokens 768 \
  --results-jsonl "$LEGAL_PROMPT_EXTRACTIVE"

echo
echo "=== Comparison summary ==="
singularity exec "${SINGULARITY_ARGS[@]}" "$CONTAINER" python -m src.compare_eval_results \
  "baseline_fixed_dense=$BASELINE_RETRIEVAL" \
  "legal_hybrid_retrieval=$LEGAL_RETRIEVAL" \
  "legal_hybrid_baseline_prompt=$LEGAL_PROMPT_BASELINE" \
  "legal_hybrid_strict_prompt=$LEGAL_PROMPT_STRICT" \
  "legal_hybrid_extractive_prompt=$LEGAL_PROMPT_EXTRACTIVE"

echo
echo "Structured result files:"
echo "- $BASELINE_RETRIEVAL"
echo "- $LEGAL_RETRIEVAL"
echo "- $LEGAL_PROMPT_BASELINE"
echo "- $LEGAL_PROMPT_STRICT"
echo "- $LEGAL_PROMPT_EXTRACTIVE"
