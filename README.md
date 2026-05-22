# Zakonodajko: Slovenian Tax RAG Assistant

This repository contains a Retrieval-Augmented Generation pipeline for answering
Slovenian tax questions from a local legal document collection.

```text
raw documents -> text extraction -> chunking -> embeddings -> FAISS retrieval -> local LLM generation
```

The target domain is Slovenian taxes, including DDV, dohodnina, tax procedure,
taxable persons, deadlines, and deductible costs.

## Repository Layout

```text
src/             Python RAG pipeline
data/raw/        local raw documents, not committed
data/processed/  generated chunks, not committed
data/index/      generated FAISS index and chunk metadata, not committed
data/eval/       generated evaluation indexes, not committed
downloads/pisrs/ PISRS HTML files used on the ARNES server
prompts/         system prompts for grounded tax answers
evaluation/      evaluation question sets
slurm/           ARNES SLURM job scripts
logs/            runtime and evaluation logs
report/          course report material
```

## 1. Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the ARNES server, use the same commands from the repository root. If a shared
Python environment is already active, the important part is that
`python -m src.rag_cli --help` works before running the full pipeline.

## 2. Model Paths

The default generation model is GaMS-9B:

```text
/d/hpc/projects/onj_fri/brainstorm/models/GaMS-9B-Instruct
```

The Mistral model is also available:

```text
/d/hpc/projects/onj_fri/models/intent
```

Because GaMS-9B is the default, ordinary commands use it automatically. To run
with Mistral instead, pass:

```bash
--model-path /d/hpc/projects/onj_fri/models/intent
```

Run answer generation on a GPU node or through a GPU SLURM job, not on a login
node.

## 3. Prepare Documents

For generic local runs, put supported documents into:

```text
data/raw/
```

Supported input formats are `.txt`, `.md`, `.pdf`, `.html`, and `.htm`.

On the ARNES server used for this project, the PISRS legal HTML files are
expected in:

```text
downloads/pisrs/
```

If the documents are somewhere else, pass that directory with `--raw-dir`.

## 4. Build The Index

Build the default local index from `data/raw/`:

```bash
python -m src.rag_cli --build-index
```

This creates:

```text
data/processed/chunks.jsonl
data/index/faiss.index
data/index/chunks.jsonl
```

On the ARNES server, build the legal index from the PISRS documents:

```bash
python -m src.rag_cli \
  --build-index \
  --raw-dir downloads/pisrs \
  --chunk-strategy legal \
  --chunk-size 1800 \
  --overlap 150 \
  --processed-chunks-path data/eval/model-compare-legal/processed_chunks.jsonl \
  --index-path data/eval/model-compare-legal/faiss.index \
  --index-chunks-path data/eval/model-compare-legal/chunks.jsonl
```

The default embedding model is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 5. Ask A Question

After building the default local index:

```bash
python -m src.rag_cli --ask "Kaj je DDV?"
```

On the ARNES server, using the shared legal index and default GaMS-9B model:

```bash
python -m src.rag_cli \
  --ask "Kaj je DDV?" \
  --index-path data/eval/model-compare-legal/faiss.index \
  --chunks-path data/eval/model-compare-legal/chunks.jsonl \
  --system-prompt prompts/tax_assistant_strict_prompt.txt \
  --retrieval-mode hybrid \
  --candidate-k 100 \
  --top-k 3 \
  --max-new-tokens 384
```

To use Mistral for the same command, add:

```bash
--model-path /d/hpc/projects/onj_fri/models/intent
```

## 6. RAG Chat

Start interactive RAG chat after building an index:

```bash
python -m src.rag_cli --chat
```

The chat keeps recent conversation turns in memory for follow-up questions, but
retrieved chunks remain the factual source for each answer. Use `/sources` to
show the chunks retrieved for the last answer, `/clear` to reset the stored
conversation context, and `/exit`, `/quit`, or Ctrl-D to stop.

On the ARNES server, use the shared legal index and default GaMS-9B model:

```bash
python -m src.rag_cli \
  --chat \
  --index-path data/eval/model-compare-legal/faiss.index \
  --chunks-path data/eval/model-compare-legal/chunks.jsonl \
  --system-prompt prompts/tax_assistant_strict_prompt.txt \
  --retrieval-mode hybrid \
  --candidate-k 100 \
  --top-k 3 \
  --max-new-tokens 384
```

For plain model chat without retrieval or source citations:

```bash
python -m src.rag_cli --direct-chat
```

## 7. Evaluation

The starter set is:

```text
evaluation/sample_questions.jsonl
```

The main evaluation set is:

```text
evaluation/tax_eval_questions.jsonl
```

Run the side-by-side retrieval and prompt evaluation on ARNES:

```bash
sbatch slurm/run_rag_eval_v2.sh
```

Run the final Mistral vs. GaMS-9B answer-generation comparison:

```bash
sbatch slurm/compare_mistral_gams9_final_v100.sh
```

Both scripts write JSONL results and SLURM output files to `logs/`.

## 8. Reproducibility Notes

- Keep raw datasets, generated chunks, FAISS indexes, logs, and large model
  files out of GitHub.
- Rebuild the index whenever the source collection or chunking settings change.
- Record raw document location, model path, embedding model, chunking strategy,
  retrieval settings, prompt file, command, and result JSONL path.
- Generation uses deterministic decoding (`do_sample=False`), but runtime can
  still vary by GPU type and installed library versions.
