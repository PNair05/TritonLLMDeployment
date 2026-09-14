"""Minimal terminal chat wrapper around the Triton endpoint."""

import argparse

from triton_client import TritonClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TritonClient(args.url)
    if not client.server_ready():
        raise RuntimeError(f"Triton or the ensemble model is not ready at {args.url}")

    print("Connected to Mistral 7B through Triton.")
    print("This first version is single-turn because the engine has a 128-token input limit.")
    print("Type quit to exit.\n")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"quit", "exit"}:
            return
        if not prompt:
            continue
        try:
            print(f"model> {client.generate(prompt, args.max_tokens)}\n")
        except RuntimeError as exc:
            print(f"error> {exc}\n")


if __name__ == "__main__":
    main()
