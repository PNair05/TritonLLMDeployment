"""Measure a reproducible Hugging Face/PyTorch baseline on one GPU.

This intentionally does not use TensorRT-LLM or Triton. It is the reference
point that later optimized measurements must match.
"""

import argparse
import statistics
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_PROMPT = (
    "Explain in three concise paragraphs how a CPU differs from a GPU "
    "when serving a transformer language model."
)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on the RunPod GPU.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Model: {args.model}")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start

    messages = [{"role": "user", "content": args.prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = args.prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_tokens = inputs["input_ids"].shape[-1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": pad_token_id,
    }

    print(f"Input tokens: {input_tokens}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Measured runs: {args.runs}")
    print("Batch size: 1")
    print("Precision: FP16")
    print(f"Loading seconds: {load_seconds:.2f}")

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model.generate(**generation_kwargs)
        cuda_sync()

        torch.cuda.reset_peak_memory_stats()
        latencies = []
        generated_tokens = []
        first_output = None

        for run_index in range(args.runs):
            cuda_sync()
            start = time.perf_counter()
            output_ids = model.generate(**generation_kwargs)
            cuda_sync()
            elapsed = time.perf_counter() - start

            output_count = output_ids.shape[-1] - input_tokens
            latencies.append(elapsed)
            generated_tokens.append(output_count)
            if first_output is None:
                first_output = output_ids
            print(
                f"Run {run_index + 1:02d}: {elapsed:.3f}s, "
                f"{output_count / elapsed:.2f} generated tokens/s"
            )

    decoded = tokenizer.decode(first_output[0][input_tokens:], skip_special_tokens=True)
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    mean_latency = statistics.mean(latencies)
    mean_generated = statistics.mean(generated_tokens)

    print("\nSummary")
    print(f"Mean latency: {mean_latency:.3f}s")
    print(f"Median latency: {statistics.median(latencies):.3f}s")
    if len(latencies) > 1:
        print(f"Latency standard deviation: {statistics.stdev(latencies):.3f}s")
    print(f"Mean generated tokens: {mean_generated:.1f}")
    print(f"Mean generation throughput: {mean_generated / mean_latency:.2f} tokens/s")
    print(f"Peak allocated GPU memory: {peak_memory_gb:.2f} GiB")
    print(f"Sample output:\n{decoded}")


if __name__ == "__main__":
    main()
