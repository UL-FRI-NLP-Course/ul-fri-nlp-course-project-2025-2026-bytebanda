#!/usr/bin/env bash
# Minimal GaMS3-12B smoke test on a V100S GPU.
#
# Submit from the project root:
#   sbatch slurm/test_gams3_v100.sh

#SBATCH --job-name=gams3-v100-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --constraint=v100s
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script must be submitted with sbatch, not run directly."
  echo "Use: sbatch slurm/test_gams3_v100.sh"
  exit 1
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONTAINER="${CONTAINER:-/d/hpc/projects/onj_fri/trije_konjeniki_apokalipse/containers/container-pytorch-rag.sif}"
MODEL="${MODEL:-/d/hpc/projects/onj_fri/dreitokens/models/GaMS3-12B-Instruct}"

cd "$PROJECT_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Project root: $PROJECT_ROOT"
echo "Container: $CONTAINER"
echo "Model: $MODEL"
echo
echo "=== GPU ==="
nvidia-smi || true

echo
echo "=== Torch CUDA check ==="
singularity exec --nv "$CONTAINER" python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
    print("vram_gb:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
    print("bf16_supported:", torch.cuda.is_bf16_supported())
PY

echo
echo "=== GaMS3-12B RAG smoke test ==="
singularity exec --nv "$CONTAINER" python -m src.rag_cli \
  --ask "Kaj je predmet obdavčitve z DDV po ZDDV-1?" \
  --model-path "$MODEL" \
  --retrieval-mode hybrid \
  --candidate-k 50 \
  --top-k 1 \
  --max-new-tokens 128
