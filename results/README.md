# Benchmark evidence

This directory keeps compact, reviewable benchmark records in Git while large
engine files and scratch logs remain ignored. The result is a controlled
same-pod comparison, not a claim about every Mistral deployment or GPU.

## Controlled workload

| Variable | PyTorch FP16 | TensorRT-LLM FP16 |
| --- | --- | --- |
| GPU | Same RTX 4090 pod | Same RTX 4090 pod |
| Model | Mistral-7B-Instruct-v0.3 | Mistral-7B-Instruct-v0.3 |
| Precision | FP16 | FP16 |
| Batch size | 1 | 1 |
| Input tokens | 28 | 28 |
| Generated tokens | Exactly 128 | Exactly 128 |
| Decoding | Greedy, forced length | Top-k 1, EOS ignored |
| Warmups / samples | 3 / 10 | 3 / 10 |
| Tokenization timed | No | No |

Both decoding configurations force the same deterministic amount of generation
work. Each latency includes prompt prefill and all 128 autoregressive decode
steps. Model or engine loading is measured separately.

## Recorded results

| Metric | PyTorch FP16 | TensorRT-LLM FP16 |
| --- | ---: | ---: |
| Mean latency | 3.924 s | 2.038 s |
| Median latency | 3.918 s | 2.037 s |
| Standard deviation | 0.026 s | 0.001 s |
| Mean generated tokens | 128.0 | 128.0 |
| Generation throughput | 32.62 tokens/s | 62.82 tokens/s |
| Load time | 26.37 s | 40.28 s |

The derived comparison is:

```text
throughput speedup = 62.82 / 32.62 = 1.93x
latency reduction  = (3.924 - 2.038) / 3.924 = 48.1%
```

Memory figures are deliberately not presented as a comparative improvement.
PyTorch recorded peak memory allocated by its allocator (13.53 GiB), whereas
TensorRT-LLM recorded total device memory in use after the benchmark (15.39
GiB). They measure different scopes.

## Evidence files and provenance

- [`pytorch-fp16-rtx4090-batch1.txt`](pytorch-fp16-rtx4090-batch1.txt) is a
  normalized transcription of the captured RunPod console record. It includes
  all ten measured request latencies and the summary emitted by `baseline.py`.
- [`tensorrt-llm-fp16-rtx4090-batch1-summary.txt`](tensorrt-llm-fp16-rtx4090-batch1-summary.txt)
  records the TensorRT-LLM summary used in the comparison. The full original
  stdout was written on RunPod to
  `outputs/trtllm-fp16-rtx4090-batch1.txt`; that ignored file has not been
  copied into this repository, so this artifact is explicitly labeled as a
  summary rather than raw output.
- [`../baseline.py`](../baseline.py) and
  [`../phase2/benchmark_engine.py`](../phase2/benchmark_engine.py) are the
  executable measurement definitions. They print the environment, workload,
  every run, and aggregate statistics.

An earlier 49.65 tokens/s TensorRT-LLM result came from a different RTX 4090
host and software environment. It is intentionally excluded from the headline
comparison.

## Reproduce or refresh the evidence

```bash
mkdir -p outputs
python -u baseline.py \
  --warmup-runs 3 --runs 10 --max-new-tokens 128 \
  | tee outputs/mistral-7b-fp16-rtx4090-current-pod-batch1.txt

python phase2/benchmark_engine.py \
  --engine-dir artifacts/trtllm/fp16-engine \
  --warmup-runs 3 --runs 10 --max-new-tokens 128 \
  | tee outputs/trtllm-fp16-rtx4090-batch1.txt
```

When publishing a refreshed result, preserve the complete stdout, record the
GPU/container versions, and update both tracked evidence files and the README
tables in the same commit.

