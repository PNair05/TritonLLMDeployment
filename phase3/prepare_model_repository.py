"""Create a Triton model repository for the Phase 2 TensorRT-LLM engine."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ENGINE = "artifacts/trtllm/fp16-engine"
DEFAULT_MODEL_REPOSITORY = "artifacts/triton/model-repository"
DEFAULT_TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_TEMPLATE_REPOSITORY = "/app/all_models/inflight_batcher_llm"
DEFAULT_FILL_TEMPLATE = "/app/tools/fill_template.py"
EXPECTED_TRTLLM_PREFIX = "0.18."
MODEL_NAMES = ("ensemble", "preprocessing", "tensorrt_llm", "postprocessing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", default=DEFAULT_ENGINE)
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_REPOSITORY)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--template-repository", default=DEFAULT_TEMPLATE_REPOSITORY)
    parser.add_argument("--fill-template", default=DEFAULT_FILL_TEMPLATE)
    parser.add_argument("--max-batch-size", type=int, default=1)
    return parser.parse_args()


def require_complete_engine(engine_dir: Path) -> None:
    required = (engine_dir / "config.json", engine_dir / "rank0.engine")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("The TensorRT-LLM engine is incomplete; missing: " + ", ".join(missing))


def installed_trtllm_version() -> str:
    result = subprocess.run(
        [sys.executable, "-c", "import tensorrt_llm; print(tensorrt_llm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def resolve_tokenizer(tokenizer: str) -> Path:
    local_path = Path(tokenizer)
    if local_path.exists():
        return local_path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=tokenizer)).resolve()


def fill_config(fill_script: Path, config: Path, substitutions: dict[str, str]) -> None:
    encoded = ",".join(f"{key}:{value}" for key, value in substitutions.items())
    subprocess.run(
        [sys.executable, str(fill_script), "-i", str(config), encoded],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.max_batch_size != 1:
        raise ValueError(
            "The current Phase 2 engine was built for max_batch_size=1. "
            "Build a serving-sized engine before choosing a larger value."
        )

    version = installed_trtllm_version()
    if not version.startswith(EXPECTED_TRTLLM_PREFIX):
        raise RuntimeError(
            f"Expected TensorRT-LLM {EXPECTED_TRTLLM_PREFIX}x, found {version}"
        )

    engine_dir = Path(args.engine_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    template_repository = Path(args.template_repository)
    fill_script = Path(args.fill_template)

    require_complete_engine(engine_dir)
    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists: {output_dir}. Remove this generated directory "
            "before intentionally rebuilding it."
        )
    if not template_repository.is_dir():
        raise RuntimeError(f"Triton model templates not found: {template_repository}")
    if not fill_script.is_file():
        raise RuntimeError(f"fill_template.py not found: {fill_script}")

    tokenizer_dir = resolve_tokenizer(args.tokenizer)
    output_dir.mkdir(parents=True)
    for model_name in MODEL_NAMES:
        source = template_repository / model_name
        if not source.is_dir():
            raise RuntimeError(f"Required Triton template is missing: {source}")
        shutil.copytree(source, output_dir / model_name)

    common = str(args.max_batch_size)
    fill_config(
        fill_script,
        output_dir / "ensemble" / "config.pbtxt",
        {"triton_max_batch_size": common, "logits_datatype": "TYPE_FP32"},
    )
    fill_config(
        fill_script,
        output_dir / "preprocessing" / "config.pbtxt",
        {
            "tokenizer_dir": str(tokenizer_dir),
            "triton_max_batch_size": common,
            "preprocessing_instance_count": "1",
            "add_special_tokens": "true",
            "engine_dir": str(engine_dir),
        },
    )
    fill_config(
        fill_script,
        output_dir / "tensorrt_llm" / "config.pbtxt",
        {
            "triton_backend": "tensorrtllm",
            "triton_max_batch_size": common,
            "decoupled_mode": "false",
            "max_beam_width": "1",
            "engine_dir": str(engine_dir),
            "max_queue_delay_microseconds": "0",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_size": "0",
            "kv_cache_free_gpu_mem_fraction": "0.20",
            "exclude_input_in_output": "true",
            "enable_kv_cache_reuse": "false",
            "encoder_input_features_data_type": "TYPE_FP16",
            "logits_datatype": "TYPE_FP32",
        },
    )
    fill_config(
        fill_script,
        output_dir / "postprocessing" / "config.pbtxt",
        {
            "tokenizer_dir": str(tokenizer_dir),
            "triton_max_batch_size": common,
            "postprocessing_instance_count": "1",
            "skip_special_tokens": "true",
        },
    )

    manifest = {
        "engine_dir": str(engine_dir),
        "max_batch_size": args.max_batch_size,
        "template_repository": str(template_repository),
        "tokenizer_dir": str(tokenizer_dir),
        "tensorrt_llm_version": version,
    }
    (output_dir / "phase3-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Created Triton model repository: {output_dir}")
    print(f"Engine: {engine_dir}")
    print(f"Tokenizer: {tokenizer_dir}")


if __name__ == "__main__":
    main()
