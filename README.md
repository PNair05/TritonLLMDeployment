# Mistral 7B inference optimization with TensorRT-LLM and Triton

This project measures and explains the effect of moving an open-weight LLM
from standard Hugging Face/PyTorch inference to TensorRT-LLM, quantizing it,
and serving the optimized model through NVIDIA Triton Inference Server.

## Current status

- [x] Establish a reproducible Hugging Face/PyTorch FP16 baseline.
- [x] Benchmark a TensorRT-LLM FP16 engine to isolate runtime/compiler gains.
- [ ] Benchmark TensorRT-LLM FP8 to isolate the effect of quantization.
- [x] Scaffold the pinned Triton model repository and endpoint client.
- [x] Serve the selected engine through Triton.
- [x] Connect a minimal application to the Triton endpoint.

## Phase 1 baseline

The baseline was measured on September 13, 2026.

| Setting | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Container | NVIDIA Triton 25.03 TensorRT-LLM |
| Framework | PyTorch 2.7.0a0 (`nv25.03`) with CUDA 12.8 |
| Model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Precision | FP16 |
| Batch size | 1 |
| Input length | 28 tokens |
| Output length | 128 tokens, forced |
| Decoding | Greedy (`do_sample=False`) |
| KV cache | Enabled |
| Warmup runs | 3 |
| Measured runs | 10 |

| Result | Value |
| --- | ---: |
| Mean latency | 3.924 s |
| Median latency | 3.918 s |
| Latency standard deviation | 0.026 s |
| Latency coefficient of variation | 0.66% |
| Mean generation throughput | 32.62 tokens/s |
| Peak allocated GPU memory | 13.53 GiB |

The latency times one complete request after tokenization and includes both
prompt prefill and autoregressive decoding. It excludes model loading. The
throughput is generated tokens divided by that end-to-end latency. This test
does not yet measure time to first token, inter-token latency, concurrent
requests, or output quality.

The TensorRT-LLM FP16 test used the same physical pod, prompt, batch size,
input length, output length, decoding strategy, warmup count, and measured-run
count. An earlier run on a different RTX 4090 host reached 49.65 tokens/s, but
it is retained only as historical evidence and is not used as the comparison
denominator because host and framework versions differed.

## Phase 2 FP16 result

| Metric | PyTorch FP16 | TensorRT-LLM FP16 | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 3.924 s | 2.038 s | 48.1% lower |
| Median latency | 3.918 s | 2.037 s | 48.0% lower |
| Latency standard deviation | 0.026 s | 0.001 s | 96.2% lower |
| Generation throughput | 32.62 tokens/s | 62.82 tokens/s | 1.93x |
| Model/engine loading | 26.37 s | 40.28 s | 52.8% longer |

TensorRT-LLM nearly doubled steady-state generation throughput and made the
measured runs more consistent, at the cost of a longer one-time engine load.
That startup cost can be amortized by a long-running inference service. GPU
memory is not compared numerically here: the baseline reports PyTorch's peak
allocator usage (13.53 GiB), while the TensorRT-LLM measurement reports total
device memory in use after execution (15.39 GiB), so they are not equivalent
metrics.

## Reproduce the baseline on RunPod

For the controlled comparison, run the baseline directly in the pinned Triton
25.03 TensorRT-LLM container used by Phase 2:

```bash
cd /workspace/TritonLLMDeployment
source scripts/runpod_env.sh
```

Run the benchmark:

```bash
mkdir -p outputs
python -u baseline.py \
  --warmup-runs 3 \
  --runs 10 \
  --max-new-tokens 128 \
  | tee outputs/mistral-7b-fp16-rtx4090-current-pod-batch1.txt
```

Files under `outputs/` are intentionally ignored by Git. Stable measurements
and their methodology belong in this README; raw logs can remain local.

## RunPod CUDA compatibility note

An earlier RunPod image used PyTorch built for CUDA 13 with a 570-series host
driver. Its CUDA forward-compatibility library failed on the consumer RTX 4090
with error 804. The pinned Triton container uses CUDA 12.8, and
`scripts/runpod_env.sh` places RunPod's host-mounted driver library before any
incompatible forward-compatibility library.

## Phase 3 serving scaffold

Phase 3 starts with a functional deployment of the validated FP16 engine before
changing engine shapes or introducing concurrent load. The setup, launch,
smoke-test, and minimal chat commands are documented in
[`phase3/README.md`](phase3/README.md). The service and chat client were
exercised successfully on the RunPod GPU on September 13, 2026.
