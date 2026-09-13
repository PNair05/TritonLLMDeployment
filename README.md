# Mistral 7B inference optimization with TensorRT-LLM and Triton

This project measures and explains the effect of moving an open-weight LLM
from standard Hugging Face/PyTorch inference to TensorRT-LLM, quantizing it,
and serving the optimized model through NVIDIA Triton Inference Server.

## Current status

- [x] Establish a reproducible Hugging Face/PyTorch FP16 baseline.
- [ ] Benchmark a TensorRT-LLM FP16 engine to isolate runtime/compiler gains.
- [ ] Benchmark TensorRT-LLM FP8 to isolate the effect of quantization.
- [ ] Serve the selected engine through Triton.
- [ ] Connect a minimal application to the Triton endpoint.

## Phase 1 baseline

The baseline was measured on September 13, 2026.

| Setting | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Host driver | 570.195.03 |
| Framework | PyTorch 2.9.1 with CUDA 12.8 |
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
| Mean latency | 2.578 s |
| Median latency | 2.559 s |
| Latency standard deviation | 0.074 s |
| Latency coefficient of variation | 2.86% |
| Mean generation throughput | 49.65 tokens/s |
| Peak allocated GPU memory | 13.53 GiB |

The latency times one complete request after tokenization and includes both
prompt prefill and autoregressive decoding. It excludes model loading. The
throughput is generated tokens divided by that end-to-end latency. This test
does not yet measure time to first token, inter-token latency, concurrent
requests, or output quality.

The TensorRT-LLM FP16 and FP8 tests will use the same prompt, batch size,
input length, output length, decoding strategy, warmup count, and measured-run
count. With a fixed output length, the speedup can be reported equivalently as
baseline latency divided by optimized latency or optimized throughput divided
by 49.65 tokens/s.

## Reproduce the baseline on RunPod

Create an isolated environment so a CUDA-incompatible PyTorch installation in
the base image cannot leak into the benchmark:

```bash
cd /workspace/TritonLLMDeployment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-baseline.txt
source scripts/runpod_env.sh
```

Run the benchmark:

```bash
mkdir -p outputs
python baseline.py \
  --warmup-runs 3 \
  --runs 10 \
  --max-new-tokens 128 \
  | tee outputs/mistral-7b-fp16-rtx4090-batch1.txt
```

Files under `outputs/` are intentionally ignored by Git. Stable measurements
and their methodology belong in this README; raw logs can remain local.

## RunPod CUDA compatibility note

The first RunPod image used PyTorch built for CUDA 13 with a 570-series host
driver. Its CUDA forward-compatibility library failed on the consumer RTX 4090
with error 804. The baseline instead uses PyTorch's CUDA 12.8 wheel and
`scripts/runpod_env.sh` places RunPod's host-mounted driver library before the
incompatible forward-compatibility library.
