#!/usr/bin/env bash
# Compare Mistral, GaMS-2B, GaMS-9B, and GaMS3-12B as RAG answer generators.
#
# Submit from the project root:
#   sbatch slurm/compare_mistral_gams9_full.sh
#
# Optional overrides:
#   QUESTIONS=evaluation/final_dataset.jsonl sbatch slurm/compare_mistral_gams9_full.sh
#   LIMIT=5 sbatch slurm/compare_mistral_gams9_full.sh
#   REBUILD_INDEX=1 sbatch slurm/compare_mistral_gams9_full.sh

#SBATCH --job-name=rag-model-compare
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch, not run directly."
  echo "Use: sbatch slurm/compare_mistral_gams9_full.sh"
  exit 1
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER="${CONTAINER:-/d/hpc/projects/onj_fri/trije_konjeniki_apokalipse/containers/container-pytorch-rag.sif}"
RUN_ID="${SLURM_JOB_ID:-manual}"

QUESTIONS="${QUESTIONS:-evaluation/final_dataset.jsonl}"
LIMIT="${LIMIT:-0}"
TOP_K="${TOP_K:-3}"
CANDIDATE_K="${CANDIDATE_K:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-prompts/tax_assistant_strict_prompt.txt}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"

MISTRAL_MODEL="${MISTRAL_MODEL:-/d/hpc/projects/onj_fri/models/intent}"
GAMS2_MODEL="${GAMS2_MODEL:-/d/hpc/projects/onj_fri/nlpmaxxing/hf/hub/models--cjvt--GaMS-2B-Instruct/snapshots/4a3fd1c568e263761ec3f7e6311732a644476316}"
GAMS9_MODEL="${GAMS9_MODEL:-/d/hpc/projects/onj_fri/brainstorm/models/GaMS-9B-Instruct}"
GAMS12_MODEL="${GAMS12_MODEL:-/d/hpc/projects/onj_fri/dreitokens/models/GaMS3-12B-Instruct}"

INDEX_DIR="${INDEX_DIR:-data/eval/model-compare-legal}"
MISTRAL_RESULTS="logs/model-compare-final-mistral-${RUN_ID}.jsonl"
GAMS2_RESULTS="logs/model-compare-final-gams2-${RUN_ID}.jsonl"
GAMS9_RESULTS="logs/model-compare-final-gams9-${RUN_ID}.jsonl"
GAMS12_RESULTS="logs/model-compare-final-gams12-${RUN_ID}.jsonl"
TIMINGS_TSV="logs/model-compare-final-timings-${RUN_ID}.tsv"

cd "$PROJECT_ROOT"
mkdir -p logs "$INDEX_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
export SINGULARITYENV_TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE"
export APPTAINERENV_TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE"
if [[ -n "${RAG_CACHE_IMPLEMENTATION:-}" ]]; then
  export SINGULARITYENV_RAG_CACHE_IMPLEMENTATION="$RAG_CACHE_IMPLEMENTATION"
  export APPTAINERENV_RAG_CACHE_IMPLEMENTATION="$RAG_CACHE_IMPLEMENTATION"
fi

echo "Project root: $PROJECT_ROOT"
echo "Container: $CONTAINER"
echo "Questions: $QUESTIONS"
if [[ "$LIMIT" -gt 0 ]]; then
  echo "Limit: $LIMIT"
else
  echo "Limit: full dataset"
fi
echo "Mistral model: $MISTRAL_MODEL"
echo "GaMS-2B model: $GAMS2_MODEL"
echo "GaMS-9B model: $GAMS9_MODEL"
echo "GaMS3-12B model: $GAMS12_MODEL"
echo "Index dir: $INDEX_DIR"
echo "Timings: $TIMINGS_TSV"
echo
echo "=== GPU ==="
nvidia-smi || true

if [[ "$REBUILD_INDEX" == "1" || ! -s "$INDEX_DIR/faiss.index" || ! -s "$INDEX_DIR/chunks.jsonl" ]]; then
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
else
  echo
  echo "=== Reusing existing shared legal index ==="
  echo "$INDEX_DIR/faiss.index"
  echo "$INDEX_DIR/chunks.jsonl"
fi

printf "label\tseconds\tminutes\tresults\n" > "$TIMINGS_TSV"

run_eval() {
  local label="$1"
  local title="$2"
  local model_path="$3"
  local results_path="$4"
  local start_ts end_ts elapsed minutes

  echo
  echo "=== Evaluating ${title} generator ==="
  echo "Model path: $model_path"
  echo "Results: $results_path"

  start_ts="$(date +%s)"
  singularity exec --nv "$CONTAINER" python -m src.evaluate_rag \
    --run-label "${label}-generator-final" \
    --run-id "$RUN_ID" \
    --prompt-label strict \
    --questions "$QUESTIONS" \
    --results-jsonl "$results_path" \
    --index-path "$INDEX_DIR/faiss.index" \
    --chunks-path "$INDEX_DIR/chunks.jsonl" \
    --model-path "$model_path" \
    --system-prompt "$SYSTEM_PROMPT" \
    --retrieval-mode hybrid \
    --candidate-k "$CANDIDATE_K" \
    --top-k "$TOP_K" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --limit "$LIMIT"
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  minutes="$(awk -v seconds="$elapsed" 'BEGIN { printf "%.2f", seconds / 60 }')"
  printf "%s\t%s\t%s\t%s\n" "$label" "$elapsed" "$minutes" "$results_path" >> "$TIMINGS_TSV"
  echo "Elapsed for ${title}: ${elapsed}s (${minutes} min)"
}

run_eval "mistral" "Mistral" "$MISTRAL_MODEL" "$MISTRAL_RESULTS"
run_eval "gams2" "GaMS-2B" "$GAMS2_MODEL" "$GAMS2_RESULTS"
run_eval "gams9" "GaMS-9B" "$GAMS9_MODEL" "$GAMS9_RESULTS"
run_eval "gams12" "GaMS3-12B" "$GAMS12_MODEL" "$GAMS12_RESULTS"

echo
echo "=== Comparison ==="
singularity exec "$CONTAINER" python -m src.compare_eval_results \
  "mistral=$MISTRAL_RESULTS" \
  "gams2=$GAMS2_RESULTS" \
  "gams9=$GAMS9_RESULTS" \
  "gams12=$GAMS12_RESULTS"

echo
echo "=== Timings ==="
cat "$TIMINGS_TSV"

echo
echo "Result files:"
echo "- $MISTRAL_RESULTS"
echo "- $GAMS2_RESULTS"
echo "- $GAMS9_RESULTS"
echo "- $GAMS12_RESULTS"
echo "- $TIMINGS_TSV"
