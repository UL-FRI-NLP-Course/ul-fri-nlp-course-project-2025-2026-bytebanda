"""Build the Version 2 RAG evaluation report from JSONL logs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "rag_evaluation_report_v2"
JOB_ID = "14625503"

RUNS = {
    "baseline_fixed_dense": ROOT / "logs" / f"rag-v2-baseline-fixed-dense-{JOB_ID}.jsonl",
    "legal_hybrid_retrieval": ROOT / "logs" / f"rag-v2-legal-hybrid-retrieval-{JOB_ID}.jsonl",
    "legal_hybrid_prompt_baseline": ROOT
    / "logs"
    / f"rag-v2-legal-hybrid-prompt-baseline-{JOB_ID}.jsonl",
    "legal_hybrid_prompt_strict": ROOT
    / "logs"
    / f"rag-v2-legal-hybrid-prompt-strict-{JOB_ID}.jsonl",
}

OLD_RUN = ROOT / "logs" / "tax-rag-eval-14619040.jsonl"
EXTRACTIVE_RUN = ROOT / "logs" / f"rag-v2-legal-hybrid-prompt-extractive-{JOB_ID}.jsonl"


MANUAL_CASES: List[Dict[str, Any]] = [
    {
        "id": "ddv_01",
        "context_relevance": 1,
        "baseline_faithfulness": 1,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 0,
        "verdict": "Retrieved VAT-related material but missed ZDDV-1 Article 3; both answers confuse taxable object with VAT amount.",
    },
    {
        "id": "ddv_02",
        "context_relevance": 1,
        "baseline_faithfulness": 2,
        "baseline_correctness": 0,
        "strict_faithfulness": 2,
        "strict_correctness": 0,
        "verdict": "Context and answers discuss VAT identification thresholds, not the general taxable-person definition in Article 5.",
    },
    {
        "id": "ddv_03",
        "context_relevance": 1,
        "baseline_faithfulness": 2,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 0,
        "verdict": "Retrieval found related VAT supply material but not Article 6; generated answers are not the legal definition of supply of goods.",
    },
    {
        "id": "ddv_04",
        "context_relevance": 1,
        "baseline_faithfulness": 2,
        "baseline_correctness": 0,
        "strict_faithfulness": 2,
        "strict_correctness": 0,
        "verdict": "The retrieved chunks concern reverse-charge and tax-event provisions, not Article 14's definition of services.",
    },
    {
        "id": "ddv_05",
        "context_relevance": 2,
        "baseline_faithfulness": 1,
        "baseline_correctness": 1,
        "strict_faithfulness": 1,
        "strict_correctness": 1,
        "verdict": "Correct Article 33 is retrieved, but generation overemphasizes payment and special intra-Union rules.",
    },
    {
        "id": "ddv_06",
        "context_relevance": 2,
        "baseline_faithfulness": 2,
        "baseline_correctness": 2,
        "strict_faithfulness": 2,
        "strict_correctness": 2,
        "verdict": "The rulebook chunk directly states 22%, 9.5%, and 5%; answers are correct despite not matching the expected statute article.",
    },
    {
        "id": "ddv_07",
        "context_relevance": 2,
        "baseline_faithfulness": 2,
        "baseline_correctness": 1,
        "strict_faithfulness": 2,
        "strict_correctness": 1,
        "verdict": "Correct identification context is present; answers cover start/request but omit changes and cessation.",
    },
    {
        "id": "doh_01",
        "context_relevance": 1,
        "baseline_faithfulness": 0,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 0,
        "verdict": "Retrieval misses Article 1; generated answers do not state that dohodnina is tax on income of natural persons.",
    },
    {
        "id": "doh_02",
        "context_relevance": 1,
        "baseline_faithfulness": 1,
        "baseline_correctness": 1,
        "strict_faithfulness": 1,
        "strict_correctness": 1,
        "verdict": "Answer partially captures resident/non-resident scope but relies on ZDDPO-2 and omits that the taxpayer is a natural person.",
    },
    {
        "id": "doh_03",
        "context_relevance": 2,
        "baseline_faithfulness": 2,
        "baseline_correctness": 2,
        "strict_faithfulness": 2,
        "strict_correctness": 2,
        "verdict": "Correct ZDoh-2 Article 8 is retrieved and both prompts answer correctly; strict prompt is cleaner.",
    },
    {
        "id": "doh_04",
        "context_relevance": 1,
        "baseline_faithfulness": 1,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 1,
        "verdict": "Retrieved chunks contain category words but not Article 18; strict answer lists many categories but frames them through exclusion text.",
    },
    {
        "id": "doh_05",
        "context_relevance": 1,
        "baseline_faithfulness": 2,
        "baseline_correctness": 0,
        "strict_faithfulness": 2,
        "strict_correctness": 0,
        "verdict": "Retrieval finds related reliefs but not Article 111; strict prompt appropriately says the exact answer is not in the context.",
    },
    {
        "id": "ddpo_01",
        "context_relevance": 0,
        "baseline_faithfulness": 1,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 0,
        "verdict": "Correct taxpayer article is not retrieved; answers incorrectly infer that the payer/izplačevalec is the DDPO taxpayer.",
    },
    {
        "id": "ddpo_02",
        "context_relevance": 1,
        "baseline_faithfulness": 2,
        "baseline_correctness": 0,
        "strict_faithfulness": 2,
        "strict_correctness": 0,
        "verdict": "Retrieved ZDDPO-2 material is about advance tax payments, not the Article 4 resident/non-resident obligation.",
    },
    {
        "id": "ddpo_03",
        "context_relevance": 1,
        "baseline_faithfulness": 1,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 0,
        "verdict": "The context mentions tax base in a receivables write-off rule; neither answer gives the Article 12 definition of tax base as profit.",
    },
    {
        "id": "ddpo_04",
        "context_relevance": 0,
        "baseline_faithfulness": 2,
        "baseline_correctness": 0,
        "strict_faithfulness": 2,
        "strict_correctness": 0,
        "verdict": "Correct rate article is not retrieved; both prompts faithfully report that the provided sources do not contain the rate.",
    },
    {
        "id": "ddpo_05",
        "context_relevance": 2,
        "baseline_faithfulness": 2,
        "baseline_correctness": 2,
        "strict_faithfulness": 2,
        "strict_correctness": 2,
        "verdict": "Correct Article 16 transfer-pricing chunks are retrieved and both answers are substantially correct.",
    },
    {
        "id": "zdavp_01",
        "context_relevance": 1,
        "baseline_faithfulness": 1,
        "baseline_correctness": 0,
        "strict_faithfulness": 1,
        "strict_correctness": 0,
        "verdict": "Retrieval misses Article 11; answers describe other tax-authority powers or procedures.",
    },
    {
        "id": "zdavp_02",
        "context_relevance": 2,
        "baseline_faithfulness": 2,
        "baseline_correctness": 1,
        "strict_faithfulness": 2,
        "strict_correctness": 1,
        "verdict": "Correct Article 12 is retrieved; answers are mostly right but omit the person liable in tax enforcement.",
    },
    {
        "id": "zdavp_03",
        "context_relevance": 2,
        "baseline_faithfulness": 2,
        "baseline_correctness": 2,
        "strict_faithfulness": 2,
        "strict_correctness": 2,
        "verdict": "Correct Article 13 is retrieved and both answers cover the right to information well.",
    },
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def code(value: Any) -> str:
    return r"\texttt{" + tex_escape(value) + "}"


def summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(records) or 1
    return {
        "n": len(records),
        "source": sum(bool(r.get("hits", {}).get("source_hit")) for r in records) / total,
        "article": sum(bool(r.get("hits", {}).get("article_hit")) for r in records) / total,
        "chunk": sum(bool(r.get("hits", {}).get("chunk_hit")) for r in records) / total,
        "phrase": sum(bool(r.get("hits", {}).get("all_phrases_hit")) for r in records) / total,
        "context": sum(r.get("scores", {}).get("context_relevance", 0) for r in records) / total,
        "faithfulness": sum(r.get("scores", {}).get("faithfulness", 0) for r in records) / total,
        "correctness": sum(r.get("scores", {}).get("answer_correctness", 0) for r in records) / total,
        "answers": sum(bool(r.get("answer")) for r in records),
        "errors": sum(bool(r.get("answer_error")) for r in records),
    }


def manual_summary() -> Dict[str, float]:
    total = len(MANUAL_CASES) or 1
    return {
        "context": sum(row["context_relevance"] for row in MANUAL_CASES) / total,
        "baseline_faithfulness": sum(row["baseline_faithfulness"] for row in MANUAL_CASES)
        / total,
        "baseline_correctness": sum(row["baseline_correctness"] for row in MANUAL_CASES)
        / total,
        "strict_faithfulness": sum(row["strict_faithfulness"] for row in MANUAL_CASES)
        / total,
        "strict_correctness": sum(row["strict_correctness"] for row in MANUAL_CASES)
        / total,
    }


def chunk_signature(record: Dict[str, Any]) -> List[tuple[str | None, str | None, str | None]]:
    return [
        (
            chunk.get("source"),
            (chunk.get("metadata") or {}).get("article_number"),
            chunk.get("chunk_id"),
        )
        for chunk in record.get("retrieved", [])
    ]


def category_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("category") or "unknown")].append(record)
    return {category: summarize(group) for category, group in sorted(groups.items())}


def format_chunks(record: Dict[str, Any]) -> str:
    parts = []
    for chunk in record.get("retrieved", []):
        metadata = chunk.get("metadata") or {}
        article = metadata.get("article_number") or "-"
        source = chunk.get("source") or "-"
        chunk_id = chunk.get("chunk_id") or "-"
        parts.append(f"{source}, {article}, {chunk_id}")
    return "; ".join(parts)


def write_manual_scores(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in MANUAL_CASES:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_tex() -> str:
    old_records = load_jsonl(OLD_RUN)
    records = {label: load_jsonl(path) for label, path in RUNS.items()}
    hybrid = {record["id"]: record for record in records["legal_hybrid_retrieval"]}
    fixed = {record["id"]: record for record in records["baseline_fixed_dense"]}
    baseline_prompt = {record["id"]: record for record in records["legal_hybrid_prompt_baseline"]}
    strict_prompt = {record["id"]: record for record in records["legal_hybrid_prompt_strict"]}
    manual_by_id = {row["id"]: row for row in MANUAL_CASES}

    summaries = {"old_v1": summarize(old_records)}
    summaries.update({label: summarize(run_records) for label, run_records in records.items()})
    manual = manual_summary()

    ids = list(hybrid)
    same_prompt_chunks = sum(
        chunk_signature(hybrid[case_id])
        == chunk_signature(baseline_prompt[case_id])
        == chunk_signature(strict_prompt[case_id])
        for case_id in ids
    )
    same_fixed_hybrid = sum(
        chunk_signature(hybrid[case_id]) == chunk_signature(fixed[case_id])
        for case_id in ids
    )

    lines: List[str] = []
    lines.extend(
        [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[english]{babel}",
            r"\usepackage{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{tabularx}",
            r"\usepackage{longtable}",
            r"\usepackage{array}",
            r"\usepackage{xcolor}",
            r"\usepackage{hyperref}",
            r"\usepackage{pdflscape}",
            r"\usepackage{enumitem}",
            r"\geometry{margin=2.2cm}",
            r"\hypersetup{colorlinks=true, linkcolor=black, urlcolor=blue}",
            r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}",
            r"\newcommand{\metric}[1]{\textbf{#1}}",
            r"\title{\textbf{Version 2 Evaluation Report for the Slovenian Tax RAG Assistant}\\\large Article-aware Chunking, Hybrid Retrieval, Prompt Comparison, and Manual Error Analysis}",
            r"\author{ByteBanda NLP Course Project}",
            r"\date{29 April 2026}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            "This report evaluates Version 2 of the Slovenian tax RAG pipeline using the JSONL logs from SLURM job "
            + code(JOB_ID)
            + ". Version 2 replaces the original fixed-only retrieval experiment with article-aware legal chunking and hybrid retrieval. "
            + "The improved retriever raises source hit@3 from 40.0\\% to 95.0\\%, article hit@3 from 0.0\\% to 30.0\\%, and exact phrase hit@3 from 0.0\\% to 35.0\\%. "
            + "Generation remains the main limitation: strict prompting is slightly more correct than the baseline prompt, but manual correctness is still only "
            + f"{manual['strict_correctness']:.2f}/2 on average because many failures originate from missing exact articles in the retrieved context.",
            r"\end{abstract}",
            r"\section{Version 2 Pipeline}",
            "The system is a Retrieval-Augmented Generation pipeline for Slovenian tax law. It ingests local PISRS HTML files, extracts visible text, chunks the documents, embeds chunks with a multilingual sentence-transformer, stores vectors in FAISS, retrieves the most similar chunks for a question, and passes those chunks to a local instruction-tuned language model.",
            r"\subsection{Document Processing and Legal Chunking}",
            "Version 1 used only overlapping fixed-size character chunks. Version 2 adds an article-aware chunking strategy in "
            + code("src/chunking.py")
            + ". The chunker detects legal headings such as "
            + code("3. člen")
            + " and "
            + code("113.a člen")
            + ", splits documents into legal sections, and attaches metadata: source filename, law identifier, document role, article number, article title, and section position. Long articles are still split into overlapping subchunks, but continuation chunks are prefixed with the law and article metadata so that embeddings retain the legal reference.",
            r"\subsection{Embedding and Similarity Search}",
            "The embedding model is "
            + code("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            + ". During indexing, every chunk text is mapped to a dense vector and normalized. FAISS "
            + code("IndexFlatIP")
            + " stores the vectors. Because vectors are normalized, inner product ranking is equivalent to cosine similarity. For a question "
            + r"$q$"
            + ", the system embeds "
            + r"$q$"
            + " into the same vector space and retrieves dense candidates by maximum cosine similarity.",
            r"\subsection{Hybrid Reranking}",
            "The Version 2 retriever first asks FAISS for a larger candidate set "
            + code("candidate_k=100")
            + " and then reranks candidates with a hybrid score:",
            r"\[",
            r"s(c, q) = s_{\mathrm{dense}}(c, q) + 0.25 \cdot s_{\mathrm{lexical}}(c, q) + s_{\mathrm{law}}(c, q)",
            r"\]",
            "where "
            + r"$s_{\mathrm{lexical}}$"
            + " is normalized token overlap between the question and chunk text plus metadata, and "
            + r"$s_{\mathrm{law}}$"
            + " is a 0.20 boost when the question names a law such as ZDDV-1, ZDoh-2, ZDDPO-2, or ZDavP-2 and the chunk metadata matches that law. The final prompt uses "
            + code("top_k=3")
            + " chunks.",
            r"\subsection{Generation}",
            "The generation step formats retrieved chunks with source filename, chunk id, score, law id, article number, and article title. Two prompt variants completed in the available logs: the original baseline prompt and a stricter prompt. The strict prompt explicitly tells the model to prefer the chunk whose legal act and article directly answer the question and to say that the answer was not found if the exact rule is missing. The extractive prompt was configured in the SLURM script but no extractive JSONL file is present for job "
            + code(JOB_ID)
            + ", so it is excluded from this report.",
            r"\section{Evaluation Design}",
            "The evaluation set contains 20 manually written tax questions. Each case includes expected source files, expected article locations, previous expected chunk IDs, key phrases, and a reference answer. Because Version 2 changes chunking, exact expected chunk IDs from Version 1 are no longer a stable primary metric. Therefore this report emphasizes source hit, article hit, phrase hit, manual context relevance, faithfulness, and correctness.",
            r"\begin{itemize}[leftmargin=*]",
            r"\item \metric{Context relevance} (0--2): whether retrieved chunks contain the exact legal rule or only related material.",
            r"\item \metric{Faithfulness} (0--2): whether the generated answer is supported by the retrieved chunks.",
            r"\item \metric{Answer correctness} (0--2): whether the generated answer matches the expected legal answer.",
            r"\end{itemize}",
            r"\section{Aggregate Results}",
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Run & n & Source@3 & Article@3 & Chunk@3 & Phrase@3 & Ctx/2 & Faith/2 & Corr/2 \\",
            r"\midrule",
        ]
    )

    run_labels = [
        ("V1 old run", "old_v1"),
        ("V2 fixed dense", "baseline_fixed_dense"),
        ("V2 legal hybrid", "legal_hybrid_retrieval"),
        ("V2 baseline prompt", "legal_hybrid_prompt_baseline"),
        ("V2 strict prompt", "legal_hybrid_prompt_strict"),
    ]
    for label, key in run_labels:
        summary = summaries[key]
        lines.append(
            f"{tex_escape(label)} & {int(summary['n'])} & "
            f"{summary['source']:.2f} & {summary['article']:.2f} & "
            f"{summary['chunk']:.2f} & {summary['phrase']:.2f} & "
            f"{summary['context']:.2f} & {summary['faithfulness']:.2f} & "
            f"{summary['correctness']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Automatic metrics from the JSONL evaluation files. Generation scores are zero for retrieval-only runs because no answers were generated.}",
            r"\end{table}",
            r"\paragraph{Manual generation assessment.}",
            "The second-pass manual assessment gives the baseline prompt an average faithfulness of "
            + f"{manual['baseline_faithfulness']:.2f}/2 and correctness of {manual['baseline_correctness']:.2f}/2. "
            + "The strict prompt gives faithfulness "
            + f"{manual['strict_faithfulness']:.2f}/2 and correctness {manual['strict_correctness']:.2f}/2. "
            + "The strict prompt is slightly better on correctness, mainly because it is more likely to refuse or narrow the answer when context is incomplete. It does not solve retrieval failures.",
            r"\paragraph{Chunk comparison.}",
            f"The retrieved top-3 chunk sets are identical between the retrieval-only, baseline-prompt, and strict-prompt legal-hybrid runs for {same_prompt_chunks}/20 cases. "
            + f"This means the prompt comparison is controlled: the model saw the same context in both prompt variants. The legal-hybrid top-3 chunks match the fixed dense top-3 chunks in {same_fixed_hybrid}/20 cases, showing that Version 2 materially changed retrieval. Exact expected chunk hit@3 remains 0/20 because expected chunk IDs were written for the previous chunking scheme; article and phrase hits are more meaningful for Version 2.",
            r"\section{Category-Level Retrieval Results}",
            r"\begin{table}[h]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Category & n & Source@3 & Article@3 & Phrase@3 & Auto ctx/2 \\",
            r"\midrule",
        ]
    )
    for category, summary in category_summary(records["legal_hybrid_retrieval"]).items():
        lines.append(
            f"{tex_escape(category)} & {int(summary['n'])} & {summary['source']:.2f} & "
            f"{summary['article']:.2f} & {summary['phrase']:.2f} & {summary['context']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Automatic retrieval metrics for the Version 2 legal-hybrid retriever, grouped by question category.}",
            r"\end{table}",
            r"\section{Per-Question Retrieval and Generation Analysis}",
            r"\begin{landscape}",
            r"\scriptsize",
            r"\begin{longtable}{p{1.7cm}p{1.2cm}p{6.0cm}p{1.3cm}p{1.4cm}p{1.4cm}p{6.0cm}}",
            r"\toprule",
            r"ID & Exp. art. & Top legal-hybrid chunks (source, article, chunk) & Ctx/2 & Base F/C & Strict F/C & Manual verdict \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"ID & Exp. art. & Top legal-hybrid chunks (source, article, chunk) & Ctx/2 & Base F/C & Strict F/C & Manual verdict \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for case_id in ids:
        record = hybrid[case_id]
        manual_row = manual_by_id[case_id]
        expected_articles = ", ".join(record.get("expected_articles") or [])
        lines.append(
            f"{code(case_id)} & {tex_escape(expected_articles)} & "
            f"{tex_escape(format_chunks(record))} & "
            f"{manual_row['context_relevance']} & "
            f"{manual_row['baseline_faithfulness']}/{manual_row['baseline_correctness']} & "
            f"{manual_row['strict_faithfulness']}/{manual_row['strict_correctness']} & "
            f"{tex_escape(manual_row['verdict'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
            r"\section{Main Findings}",
            r"\begin{enumerate}[leftmargin=*]",
            r"\item Article-aware chunking improves metadata quality and makes evaluation more meaningful. Most retrieved chunks now carry law and article information.",
            r"\item Hybrid retrieval greatly improves source-level retrieval, but it still often retrieves a related article instead of the exact definition article. This is the dominant error source.",
            r"\item Exact chunk ID matching is no longer a useful success criterion across pipeline versions, because legal chunking creates different chunk IDs. Source, article, and expected phrase coverage should be used instead.",
            r"\item Generation is usually faithful to retrieved context, including wrong context. This is useful diagnostically: answer failures mostly follow retrieval failures, not arbitrary hallucination.",
            r"\item Strict prompting improves concise behavior and some correctness cases, but cannot repair missing context. It sometimes answers in English even for Slovenian questions, so language control should be tightened.",
            r"\end{enumerate}",
            r"\section{Recommendations for the Next Iteration}",
            r"\begin{itemize}[leftmargin=*]",
            r"\item Add explicit article-number matching when the question contains or implies a known article, and include legal synonyms such as predmet obdavčitve, davčni zavezanec, davčna osnova, and stopnja davka.",
            r"\item Add BM25 or another sparse retriever and combine it with dense retrieval. Current lexical overlap helps but is too weak for exact legal definitions.",
            r"\item Index article titles separately and boost title matches. Many evaluation questions ask for the article title concept directly.",
            r"\item Rebuild the evaluation expected chunks for the legal chunking scheme, or replace chunk-id expectations with source+article+phrase checks.",
            r"\item Keep strict prompting as the default generation prompt, but add an explicit instruction to answer in Slovenian for Slovenian questions and to avoid citing non-answer chunks.",
            r"\end{itemize}",
            r"\section{Reproducibility}",
            "The evaluated JSONL files are:",
            r"\begin{itemize}[leftmargin=*]",
        ]
    )
    for label, path in RUNS.items():
        lines.append(r"\item " + code(label) + ": " + code(path.relative_to(ROOT)))
    lines.append(
        r"\item "
        + code("extractive prompt")
        + ": not present as "
        + code(EXTRACTIVE_RUN.relative_to(ROOT))
        + " in this run."
    )
    lines.extend(
        [
            r"\end{itemize}",
            "The report was generated by "
            + code("rag_evaluation_report_v2/build_report_v2.py")
            + ". Manual scores are written to "
            + code("rag_evaluation_report_v2/manual_generation_scores_v2.jsonl")
            + ".",
            r"\end{document}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_manual_scores(REPORT_DIR / "manual_generation_scores_v2.jsonl")
    tex = build_tex()
    (REPORT_DIR / "rag_evaluation_report_v2.tex").write_text(tex, encoding="utf-8")
    print(REPORT_DIR / "rag_evaluation_report_v2.tex")
    print(REPORT_DIR / "manual_generation_scores_v2.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
