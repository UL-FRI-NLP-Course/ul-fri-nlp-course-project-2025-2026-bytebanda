from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classla_support import annotate_records
from .text_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CLASSLA helpers for Zakonodajko")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate_parser = subparsers.add_parser("annotate-jsonl", help="Annotate JSONL records with CLASSLA")
    annotate_parser.add_argument("--input", type=Path, required=True)
    annotate_parser.add_argument("--output", type=Path, required=True)

    lemmatize_parser = subparsers.add_parser("lemmatize-text", help="Lemmatize a single query string")
    lemmatize_parser.add_argument("--text", type=str, required=True)

    args = parser.parse_args()
    if args.command == "annotate-jsonl":
        rows = read_jsonl(args.input)
        annotated = annotate_records(rows)
        write_jsonl(args.output, annotated)
    elif args.command == "lemmatize-text":
        annotated = annotate_records([{"unit_id": "__query__", "content_text": args.text}])[0]
        lemmas = [
            token.get("lemma", token.get("text", "")).lower()
            for sentence in annotated["sentences"]
            for token in sentence["tokens"]
            if token.get("lemma") or token.get("text")
        ]
        print(json.dumps({"lemmas": lemmas}, ensure_ascii=False))


if __name__ == "__main__":
    main()
