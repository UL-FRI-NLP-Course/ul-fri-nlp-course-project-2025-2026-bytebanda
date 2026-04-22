from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOAD_REPORT = REPO_ROOT / "downloads" / "pisrs" / "download_report.json"
DEFAULT_FURS_GUIDANCE_URL = "https://www.fu.gov.si/navodila_pojasnila_in_smernice"
DEFAULT_FURS_DDV_TOPIC_URL = "https://www.fu.gov.si/davki_in_druge_dajatve/podrocja/davek_na_dodano_vrednost_ddv/"
DEFAULT_FURS_DOWNLOAD_DIR = REPO_ROOT / "downloads" / "furs"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "retrieval"
DEFAULT_PARSED_PATH = DEFAULT_ARTIFACT_DIR / "parsed_documents.jsonl"
DEFAULT_UNIT_PATH = DEFAULT_ARTIFACT_DIR / "annotation_units.jsonl"
DEFAULT_ANNOTATION_PATH = DEFAULT_ARTIFACT_DIR / "classla_annotations.jsonl"
DEFAULT_CHUNK_PATH = DEFAULT_ARTIFACT_DIR / "chunks.jsonl"
DEFAULT_BM25_PATH = DEFAULT_ARTIFACT_DIR / "bm25_corpus.json"
DEFAULT_CHROMA_DIR = DEFAULT_ARTIFACT_DIR / "chroma"
DEFAULT_EVAL_PATH = REPO_ROOT / "dataset" / "pisrs" / "evaluation_queries.jsonl"
DEFAULT_REAL_EVAL_PATH = REPO_ROOT / "dataset" / "pisrs" / "real_eval_questions.jsonl"
DEFAULT_REAL_EVAL_DOWNLOAD_DIR = REPO_ROOT / "downloads" / "furs" / "eval"

DEFAULT_SPLIT_TRIGGER_CHARS = 2200
DEFAULT_MAX_CHUNK_CHARS = 3000
DEFAULT_OVERLAP_CHARS = 220
DEFAULT_FURS_MIN_YEAR = 2024
DEFAULT_FURS_PORTAL_MAX_PAGES = 24
DEFAULT_EMBEDDING_PROFILE = "bge_m3"
DEFAULT_LOCAL_GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LOCAL_GENERATOR_MAX_NEW_TOKENS = 256
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60

SECTION_KEYS = ("del", "poglavje", "oddelek", "pododdelek", "odsek")

ORDINAL_WORDS = (
    "prvi",
    "drugi",
    "tretji",
    "četrti",
    "peti",
    "šesti",
    "sedmi",
    "osmi",
    "deveti",
    "deseti",
)
