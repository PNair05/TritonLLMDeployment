"""Convert Mistral shard-by-shard, then build a benchmark-sized FP16 engine."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_CHECKPOINT = "artifacts/trtllm/fp16-checkpoint"
DEFAULT_ENGINE = "artifacts/trtllm/fp16-engine"
DEFAULT_CONVERTER = "/app/examples/llama/convert_checkpoint.py"
EXPECTED_TRTLLM_PREFIX = "0.18."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_ENGINE)
    parser.add_argument("--converter", default=DEFAULT_CONVERTER)
    parser.add_argument("--cuda-arch", default="8.9")
    return parser.parse_args()


def directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def checkpoint_is_complete(path: Path) -> bool:
    required = (path / "config.json", path / "rank0.safetensors")
    return all(item.is_file() and item.stat().st_size > 0 for item in required)


def engine_is_complete(path: Path) -> bool:
    required = (path / "config.json", path / "rank0.engine")
    return all(item.is_file() and item.stat().st_size > 0 for item in required)


def installed_trtllm_version() -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tensorrt_llm; print(tensorrt_llm.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def resolve_model(model: str) -> str:
    local_path = Path(model)
    if local_path.exists():
        return str(local_path.resolve())
    return snapshot_download(repo_id=model)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    engine_dir = Path(args.output_dir)
    converter = Path(args.converter)

    version = installed_trtllm_version()
    if not version.startswith(EXPECTED_TRTLLM_PREFIX):
        raise RuntimeError(
            f"Expected TensorRT-LLM {EXPECTED_TRTLLM_PREFIX}x, found {version}"
        )
    if not converter.is_file():
        raise RuntimeError(f"TensorRT-LLM converter not found: {converter}")

    if engine_is_complete(engine_dir):
        print(f"Complete engine already exists: {engine_dir}")
        return
    if directory_has_files(engine_dir):
        raise RuntimeError(
            f"Refusing to use incomplete engine directory: {engine_dir}"
        )

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    engine_dir.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_is_complete(checkpoint_dir):
        print(f"Reusing complete TensorRT-LLM checkpoint: {checkpoint_dir}")
    elif directory_has_files(checkpoint_dir):
        raise RuntimeError(
            f"Refusing to use incomplete checkpoint directory: {checkpoint_dir}"
        )
    else:
        model_dir = resolve_model(args.model)
        convert_command = [
            sys.executable,
            "-u",
            str(converter),
            "--model_dir",
            model_dir,
            "--output_dir",
            str(checkpoint_dir),
            "--dtype",
            "float16",
            "--tp_size",
            "1",
            "--workers",
            "1",
            "--load_by_shard",
            "--load_model_on_cpu",
        ]
        print("Converting checkpoint shard-by-shard:", flush=True)
        print(shlex.join(convert_command), flush=True)
        subprocess.run(convert_command, check=True)
        if not checkpoint_is_complete(checkpoint_dir):
            raise RuntimeError("Checkpoint conversion exited without complete files")

    build_executable = shutil.which("trtllm-build")
    if build_executable is None:
        raise RuntimeError("trtllm-build was not found on PATH")

    # Replace this Python process with the builder so conversion-time memory is
    # fully released before TensorRT begins its memory-intensive build phase.
    build_command = [
        build_executable,
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(engine_dir),
        "--max_batch_size",
        "1",
        "--max_input_len",
        "128",
        "--max_seq_len",
        "256",
        "--max_num_tokens",
        "256",
        "--opt_num_tokens",
        "128",
        "--gemm_plugin",
        "float16",
        "--workers",
        "1",
    ]
    environment = os.environ.copy()
    environment.setdefault("TORCH_CUDA_ARCH_LIST", args.cuda_arch)
    print("Building TensorRT-LLM engine in a fresh process:", flush=True)
    print(shlex.join(build_command), flush=True)
    os.execvpe(build_command[0], build_command, environment)


if __name__ == "__main__":
    main()
