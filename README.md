# Zakonodajko: Slovenian Tax RAG Assistant

This repository contains a minimal Retrieval-Augmented Generation baseline for
the NLP 2026 laboratory project. The assistant answers Slovenian tax questions
from a local document collection instead of relying only on model memory.

The current implementation is intentionally simple and reproducible:

```text
raw docs -> text extraction -> chunking -> embeddings -> FAISS retrieval -> Mistral generation
```

The target domain is Slovenian taxes, including topics such as DDV, dohodnina,
tax procedure, taxable persons, deadlines, and deductible costs.

## Repository Layout

```text
src/          Python RAG pipeline
data/raw/     raw local documents, not committed
data/processed/ generated chunks, not committed
data/index/   generated FAISS index and chunk metadata, not committed
prompts/      system prompt for grounded tax answers
evaluation/   sample evaluation questions
slurm/        ARNES HPC job scripts
logs/         SLURM/runtime logs, not committed
report/       course report material
```

## Dataset Placement

Put raw tax documents into:

```text
data/raw/
```

Supported input formats are `.txt`, `.md`, `.pdf`, `.html`, and `.htm`.
The pipeline extracts text, keeps source filename metadata, keeps PDF page
numbers when available, and writes chunks to `data/processed/chunks.jsonl`.

Do not hardcode local machine paths such as Downloads. On the ARNES cluster,
copy or sync the dataset into `~/tax_project/data/raw/`.

## Setup

Create an environment with Python and install the dependencies:

```bash
pip install -r requirements.txt
```

On the ARNES HPC with the provided Singularity container:

```bash
singularity exec ~/containers/pytorch.sif pip install -r requirements.txt
```

The local generation model is expected at:

```text
/d/hpc/projects/onj_fri/models/intent
```

This path is used as the default Mistral-7B-Instruct-v0.3 model path.
For full answer generation, run through an interactive or batch HPC job rather
than a login node.

## Build The Index

After placing source documents in `data/raw/`, build the FAISS index:

```bash
python -m src.rag_cli --build-index
```

This creates:

```text
data/processed/chunks.jsonl
data/index/faiss.index
data/index/chunks.jsonl
```

The default embedding model is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Ask A Question

```bash
python -m src.rag_cli --ask "Kaj je DDV?"
```

Optional retrieval depth:

```bash
python -m src.rag_cli --ask "Kaj je DDV?" --top-k 5
```

The generator receives the retrieved context, the user question, and the system
prompt in `prompts/tax_assistant_system_prompt.txt`. It is instructed to answer
only from retrieved context, cite filenames and chunk IDs, avoid definitive legal
advice, and say when the answer is not found.

## RAG Chat

For an interactive chat where every turn retrieves tax-law sources before
generation, start:

```bash
python -m src.rag_cli --chat
```

The chat keeps recent conversation history in memory for follow-up questions,
but retrieved chunks remain the factual source for each answer. Use `/sources`
to show the chunks retrieved for the last answer, `/clear` to reset the chat
history, and `/exit`, `/quit`, or Ctrl-D to stop.

For the old plain conversational mode without retrieval or source citations,
use:

```bash
python -m src.rag_cli --direct-chat
```

## SLURM

Submit the minimal HPC smoke test from the project root:

```bash
sbatch slurm/run_rag_test.sh
```

The script uses:

```text
~/containers/pytorch.sif
```

It runs:

```bash
python -m src.rag_cli --build-index
python -m src.rag_cli --ask "Kaj je DDV?"
```

GPU partition and GPU resource lines are included as comments in the script so
they can be adapted to the active ARNES queue policy. The script writes logs to
`logs/`.

## Evaluation

`evaluation/sample_questions.jsonl` contains five starter Slovenian tax
questions for manual testing and future evaluation of factuality and relevance.
For Submission 2, these can be used to record retrieved sources, generated
answers, and observed failure modes.

The main evaluation set is `evaluation/tax_eval_questions.jsonl`. To run the
new side-by-side evaluation on the HPC, submit:

```bash
sbatch slurm/run_rag_eval_v2.sh
```

The script builds two indexes:

```text
baseline: fixed character chunks + dense FAISS retrieval
new:      article-aware legal chunks + dense candidates reranked with lexical/source signals
```

It writes separate JSONL logs for baseline retrieval, improved retrieval, and
three prompt variants. The end of the SLURM `.out` file prints a comparison
table with source hit, article hit, phrase hit, context relevance, faithfulness,
and answer correctness scores.

## Reproducibility Notes

- Keep raw datasets, generated chunks, FAISS indexes, logs, containers, and
  large model files out of GitHub.
- Rebuild the index from `data/raw/` whenever the source collection changes.
- Record model paths, embedding model names, commands, and evaluation results in
  the report so another user can reproduce the baseline.
- This baseline avoids LangChain and other larger frameworks to keep the system
  easy to inspect for the course submission.
