"""Benchmark a saved TensorRT-LLM engine using the Phase 1 workload."""

import argparse
import statistics
import time

import tensorrt_llm
import torch
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from transformers import AutoTokenizer


DEFAULT_ENGINE = "artifacts/trtllm/fp16-engine"
DEFAULT_TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_PROMPT = (
    "Explain in three concise paragraphs how a CPU differs from a GPU "
    "when serving a transformer language model."
)
EXPECTED_TRTLLM_PREFIX = "0.18."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", default=DEFAULT_ENGINE)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def device_memory_gib() -> float:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (total_bytes - free_bytes) / (1024**3)


def main() -> None:
    args = parse_args()

    if not tensorrt_llm.__version__.startswith(EXPECTED_TRTLLM_PREFIX):
        raise RuntimeError(
            f"Expected TensorRT-LLM {EXPECTED_TRTLLM_PREFIX}x, "
            f"found {tensorrt_llm.__version__}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on the RunPod GPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][
        0
    ].tolist()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TensorRT-LLM: {tensorrt_llm.__version__}")
    print(f"Engine: {args.engine_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Input tokens: {len(input_ids)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Measured runs: {args.runs}")
    print("Batch size: 1")

    sampling_params = SamplingParams(
        end_id=-1,
        pad_id=-1,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        top_k=1,
        ignore_eos=True,
        detokenize=False,
    )

    load_started = time.perf_counter()
    # TensorRT-LLM preallocates a KV-cache block pool. Ten percent of remaining
    # memory is ample for this 156-token, batch-one workload and avoids making
    # the engine appear to consume all otherwise-free VRAM.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.10)
    with LLM(
        model=args.engine_dir,
        skip_tokenizer_init=True,
        kv_cache_config=kv_cache_config,
    ) as llm:
        load_seconds = time.perf_counter() - load_started

        for _ in range(args.warmup_runs):
            llm.generate(input_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()

        latencies = []
        generated_tokens = []
        first_output_ids = None

        for run_index in range(args.runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = llm.generate(input_ids, sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            output_ids = output.outputs[0].token_ids
            output_count = len(output_ids)
            if output_count != args.max_new_tokens:
                raise RuntimeError(
                    f"Expected {args.max_new_tokens} output tokens, got {output_count}"
                )
            latencies.append(elapsed)
            generated_tokens.append(output_count)
            if first_output_ids is None:
                first_output_ids = output_ids
            print(
                f"Run {run_index + 1:02d}: {elapsed:.3f}s, "
                f"{output_count / elapsed:.2f} generated tokens/s"
            )

        device_memory = device_memory_gib()

    mean_latency = statistics.mean(latencies)
    mean_generated = statistics.mean(generated_tokens)
    decoded = tokenizer.decode(first_output_ids, skip_special_tokens=True)

    print("\nSummary")
    print(f"Engine loading seconds: {load_seconds:.2f}")
    print(f"Mean latency: {mean_latency:.3f}s")
    print(f"Median latency: {statistics.median(latencies):.3f}s")
    if len(latencies) > 1:
        print(f"Latency standard deviation: {statistics.stdev(latencies):.3f}s")
    print(f"Mean generated tokens: {mean_generated:.1f}")
    print(f"Mean generation throughput: {mean_generated / mean_latency:.2f} tokens/s")
    print(f"Device memory in use after benchmark: {device_memory:.2f} GiB")
    print(f"Sample output:\n{decoded}")


if __name__ == "__main__":
    main()
