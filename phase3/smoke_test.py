"""Check Triton readiness and send one deterministic Mistral request."""

import argparse

from triton_client import TritonClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--prompt",
        default="In two sentences, explain what an inference server does.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TritonClient(args.url)
    if not client.server_ready():
        raise RuntimeError(f"Triton or the ensemble model is not ready at {args.url}")
    print("Triton server: ready")
    print("Ensemble model: ready")
    print("\nResponse")
    print(client.generate(args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
