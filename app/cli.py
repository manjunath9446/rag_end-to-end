from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .pipeline import RAGPipeline


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_where(pairs: list[str] | None) -> dict | None:
    if not pairs:
        return None
    where: dict = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(
                f"--where expects KEY=VALUE, got {raw!r}. "
                f"Example: --where filename=acme-docs.md"
            )
        key, value = raw.split("=", 1)
        where[key.strip()] = _coerce(value.strip())
    return where


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload", help="Ingest a file")
    p_up.add_argument("path")

    p_q = sub.add_parser("retrieve", help="Query the store")
    p_q.add_argument("query")
    p_q.add_argument("--top-k", type=int, default=None)
    p_q.add_argument("--where", action="append", metavar="KEY=VALUE")

    p_a = sub.add_parser("ask", help="Ask a grounded question via Grok")
    p_a.add_argument("question")
    p_a.add_argument("--top-k", type=int, default=None)
    p_a.add_argument("--where", action="append", metavar="KEY=VALUE")

    sub.add_parser("stats")

    args = parser.parse_args(argv)
    pipeline = RAGPipeline(get_settings())

    if args.cmd == "upload":
        print(json.dumps(pipeline.ingest_file(Path(args.path)), indent=2))
    elif args.cmd == "retrieve":
        where = _parse_where(args.where)
        print(json.dumps(
            pipeline.retrieve(args.query, top_k=args.top_k, where=where),
            indent=2,
        ))
    elif args.cmd == "ask":
        where = _parse_where(args.where)
        try:
            result = pipeline.ask(args.question, top_k=args.top_k, where=where)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
    elif args.cmd == "stats":
        print(json.dumps(pipeline.stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())