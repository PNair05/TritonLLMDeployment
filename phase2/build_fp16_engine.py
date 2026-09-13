"""Build and save a benchmark-sized TensorRT-LLM FP16 engine."""

import argparse
import time
from pathlib import Path

import tensorrt_llm
from tensorrt_llm import BuildConfig, LLM


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OUTPUT = "artifacts/trtllm/fp16-engine"
DEFAULT_WORKSPACE = "/workspace/trtllm-build-workspace"
EXPECTED_TRTLLM_PREFIX = "0.18."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not tensorrt_llm.__version__.startswith(EXPECTED_TRTLLM_PREFIX):
        raise RuntimeError(
            f"Expected TensorRT-LLM {EXPECTED_TRTLLM_PREFIX}x, "
            f"found {tensorrt_llm.__version__}"
        )

    output_dir = Path(args.output_dir)
    workspace = Path(args.workspace)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty engine directory: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    # This engine is intentionally bounded to the controlled benchmark. A
    # production-serving engine will use a larger batch and sequence envelope.
    build_config = BuildConfig(
        max_input_len=128,
        max_seq_len=256,
        opt_batch_size=1,
        max_batch_size=1,
        opt_num_tokens=128,
        max_num_tokens=256,
    )

    print(f"TensorRT-LLM: {tensorrt_llm.__version__}")
    print(f"Model: {args.model}")
    print("Precision: FP16")
    print(f"Output directory: {output_dir}")

    started = time.perf_counter()
    with LLM(
        model=args.model,
        dtype="float16",
        build_config=build_config,
        workspace=str(workspace),
    ) as llm:
        llm.save(str(output_dir))
    elapsed = time.perf_counter() - started

    print(f"Engine build and save completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
