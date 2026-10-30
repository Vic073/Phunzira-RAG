from __future__ import annotations

import argparse

from .pipeline import build_assistant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phunzira", description="Chichewa/English study assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=None, help="number of passages to retrieve")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "ask":
        answer = build_assistant().ask(args.question, k=args.k)
        print(answer.text)
        if answer.sources:
            print("\nSources:")
            for s in answer.sources:
                print(f"  [{s.index}] {s.title} ({s.doc_id}, {s.lang})")
        return 0

    import uvicorn

    uvicorn.run("phunzira.api:create_app", factory=True, host=args.host, port=args.port)
    return 0
