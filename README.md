# Natural language processing course: `Zakonodajko`

![Zakonodajko](images/zakonodajko.png)

## 🦉 Zokonodajko – Your Slovenian Legal AI Assistant

**Zokonodajko** is a domain-specific conversational AI assistant designed to provide accurate, context-aware answers about Slovenian legislation. Unlike general-purpose language models, Zokonodajko is grounded in curated legal sources, enabling it to deliver reliable and explainable responses with minimal hallucination.

---

## Purpose

Zokonodajko helps users navigate complex legal information by:

- answering questions about Slovenian laws (e.g., taxes, procedures, rights)
- explaining legal concepts in simple terms
- guiding users through administrative processes

---

## How it works

The system is built using a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **User query understanding** – interprets the user’s intent  
2. **Document retrieval** – searches a curated legal knowledge base  
3. **Context injection** – provides relevant legal excerpts to the model  
4. **Answer generation** – produces grounded, explainable responses  

---

## Knowledge base

## 📚 Knowledge base

Zokonodajko relies on trusted Slovenian legal sources, including:

- Uradni list Republike Slovenije (official laws)
- Pravni informacijski sistem Republike Slovenije – primary legal texts:
  - **[Zakon o davku na dodano vrednost (ZDDV‑1)](https://pisrs.si/pregledPredpisa?id=ZAKO4701)** – the Value Added Tax law  
  - **[Zakon o dohodnini (ZDoh‑2)](https://pisrs.si/pregledPredpisa?id=ZAKO4697)** – the Income Tax Act  
  - **[Zakon o davčnem postopku (ZDavP‑2)](https://pisrs.si/pregledPredpisa?id=ZAKO4703)** – the Tax Procedure Act
- Finančna uprava Republike Slovenije (tax guidelines and explanations)
- GOV.SI (public service procedures)


---


## Example queries

- “What taxes does an s.p. pay in Slovenia?”  
- “When is the deadline for dohodnina?”  
- “What are my rights as a tenant?”  

---

## ⚠️ Disclaimer

Zokonodajko is an informational tool and does not replace professional legal advice.

---

## Project Structure

The project is organized as follows:

- **`images/`** – contains the Zokonodajko logo and other visual assets  
- **`code/`** – contains all scripts, notebooks, and code used to build the agent  
- **`dataset/`** – contains the Slovenian legal documents, PDFs, and processed text used for retrieval and embeddings

## Retrieval Baseline

The repository now includes a local PISRS + FURS ingestion and hybrid retrieval baseline under [code/zakonodajko_rag](/Users/maticko/DataScience/NLP/ul-fri-nlp-course-project-2025-2026-bytebanda/code/zakonodajko_rag).

It implements:

- HTML parsing for the current 7 downloaded PISRS tax acts and pravilniki
- indexed ingestion of public FURS guidance from `Navodila, pojasnila in smernice`
- structure-aware normalization into articles, sections, notes, tables, and attachment references
- Slovene NLP preprocessing via `classla`
- regex extraction for legal references, dates, euro amounts, and percentages
- hybrid retrieval with BM25 over lemmatized text plus dense retrieval using `BAAI/bge-m3`
- grounded answering with citation output on top of the retriever
- lokalni generativni odgovor z modelom `Qwen/Qwen2.5-1.5B-Instruct`
- lokalni web chat UI za pogovor v brskalniku

## Setup

Main environment:

```bash
python3 -m venv .venv-rag
source .venv-rag/bin/activate
python3 -m pip install -e . pytest
```

Use a dedicated project environment. Reusing a broad ML environment can cause binary conflicts, especially around `numpy`, `torch`, and unrelated packages such as `tensorflow` or `mediapipe`.

Separate `classla` environment:

```bash
python3 -m venv .venv-classla
.venv-classla/bin/python -m pip install --upgrade pip
.venv-classla/bin/python -m pip install -r requirements-classla.txt
```

## Build The Index

Run the full parser, Slovene annotation, chunking, BM25 build, and dense indexing pipeline:

```bash
python3 -m zakonodajko_rag build-all --classla-python .venv-classla/bin/python
```

Za eksplicitno izbiro dense embedding profila:

```bash
python3 -m zakonodajko_rag build-all \
  --classla-python .venv-classla/bin/python \
  --embedding-profile bge_m3
```

To include public FURS explanations and guidance from 2024 onward:

```bash
python3 -m zakonodajko_rag build-all \
  --classla-python .venv-classla/bin/python \
  --include-furs \
  --furs-min-year 2024
```

`--include-furs` zdaj poleg `Navodila, pojasnila in smernice` pobere tudi izbrane javne FURS DDV tehnične dokumente ter eDavki/OpenPortal HTML strani za evidence, oddajo, XML/CSV, tehnične specifikacije in povezane postopke. Privzeto je iz builda izključen isti `Kratka vprašanja in odgovori` dokument, iz katerega je zgrajen `real_eval_questions.jsonl`, da ostane ta eval set pošten holdout.

Generated artifacts are written to `artifacts/retrieval/`.

## Query

```bash
python3 -m zakonodajko_rag query "Kaj določa 395. člen ZDavP-2?" --classla-python .venv-classla/bin/python
```

Za query z vključenim rerankerjem:

```bash
python3 -m zakonodajko_rag query "Kako FURS razlaga DDV pri restavracijskih storitvah?" \
  --classla-python .venv-classla/bin/python \
  --reranker-model BAAI/bge-reranker-v2-m3
```

## Answer

Privzeto `answer` uporabi lokalni instruct model in ga ob prvem zagonu prenese v lokalni Hugging Face cache, če ga še nimaš:

```bash
python3 -m zakonodajko_rag answer "Kaj določa 395. člen ZDavP-2?" --classla-python .venv-classla/bin/python
```

Za manj zmogljiv računalnik lahko uporabiš manjši model:

```bash
python3 -m zakonodajko_rag answer "Kaj pomeni davčna tajnost?" \
  --classla-python .venv-classla/bin/python \
  --generator-model Qwen/Qwen2.5-0.5B-Instruct
```

Če želiš izklopiti generativni model in uporabiti samo extractive odgovor:

```bash
python3 -m zakonodajko_rag answer "Kaj določa 395. člen ZDavP-2?" \
  --classla-python .venv-classla/bin/python \
  --extractive-only
```

Za alternativni dense embedding profil:

```bash
python3 -m zakonodajko_rag answer "Katere so stopnje dohodnine?" \
  --classla-python .venv-classla/bin/python \
  --embedding-profile e5_large_instruct \
  --extractive-only
```

Če si vse namestil v en environment, ki že vsebuje `classla`, lahko `--classla-python` izpustiš.

Optional local Hugging Face generation on top of retrieved context:

```bash
python3 -m zakonodajko_rag answer "Kaj določa 395. člen ZDavP-2?" \
  --classla-python .venv-classla/bin/python \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct
```

## Web Chat

Zaženi lokalni web strežnik:

```bash
python3 -m zakonodajko_rag serve --classla-python .venv-classla/bin/python --port 8000
```

Nato odpri:

```text
http://127.0.0.1:8000
```

Frontend je lokalna chat aplikacija nad istim retrieval/answer pipeline-om. Za follow-up vprašanja uporablja osnovno kontekstualizacijo zadnjih uporabniških vprašanj, zato je še vedno najbolje, da so vprašanja čim bolj konkretna.
Pri strogo številčnih odgovorih, členih in rokih sistem raje obdrži deterministični odgovor, pri bolj razlagalnih vprašanjih pa uporabi lokalni model, če je izhod dovolj kakovosten.

Privzeto web chat uporablja isti lokalni generativni model kot `answer`. Če ga želiš izklopiti:

```bash
python3 -m zakonodajko_rag serve --classla-python .venv-classla/bin/python --extractive-only
```

Web app zna zdaj uporabiti tudi izbrani embedding profil in reranker:

```bash
python3 -m zakonodajko_rag serve \
  --classla-python .venv-classla/bin/python \
  --embedding-profile bge_m3 \
  --reranker-model BAAI/bge-reranker-v2-m3
```

## Evaluate

The committed evaluation set is stored in [dataset/pisrs/evaluation_queries.jsonl](/Users/maticko/DataScience/NLP/ul-fri-nlp-course-project-2025-2026-bytebanda/dataset/pisrs/evaluation_queries.jsonl).

```bash
python3 -m zakonodajko_rag evaluate --classla-python .venv-classla/bin/python
```

A/B primerjava embedding profilov:

```bash
python3 -m zakonodajko_rag compare-embeddings \
  --classla-python .venv-classla/bin/python \
  --embedding-profiles bge_m3 e5_large_instruct
```

## Real-World Eval From FURS Q&A

Repository now also supports a second eval track built from public FURS `Kratka vprašanja in odgovori` and `Vprašanja in odgovori` documents.

Zgradi eval set iz javno objavljenih FURS vprašanj:

```bash
python3 -m zakonodajko_rag build-real-eval \
  --output-path dataset/pisrs/real_eval_questions.jsonl \
  --download-dir downloads/furs/eval \
  --limit 60
```

Nato poženi answer-level eval nad tem setom:

```bash
python3 -m zakonodajko_rag evaluate-real \
  --classla-python .venv-classla/bin/python \
  --real-eval-path dataset/pisrs/real_eval_questions.jsonl \
  --extractive-only
```

Ta eval meri bolj praktične lastnosti kot osnovni retrieval benchmark:

- ali je glavni vir pravilen
- ali sistem zazna pravi zakon/člen, kadar je to razvidno iz odgovora
- ali je odgovor vsebinsko podoben referenčnemu FURS odgovoru
- ali je odgovor vsaj heuristično uporaben
