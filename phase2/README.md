# Phase 2: TensorRT-LLM comparison

Phase 2 separates two effects:

1. PyTorch FP16 to TensorRT-LLM FP16 measures the effect of the optimized
   engine and runtime while keeping numerical precision constant.
2. TensorRT-LLM FP16 to TensorRT-LLM FP8 measures the incremental effect of
   quantization.

Both engine benchmarks reuse Phase 1's exact tokenized prompt, batch size,
128-token output length, greedy decoding, three warmups, and ten measured runs.

## Pinned software stack

Use the NVIDIA Triton 25.03 TensorRT-LLM container:

```text
nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3
```

It contains CUDA 12.8.1, TensorRT 10.9, and TensorRT-LLM 0.18.0. CUDA 12.8
is compatible with the RunPod host's 570-series driver. Newer TensorRT-LLM
containers use CUDA 13 and cannot use CUDA forward compatibility on the
consumer RTX 4090 in this pod.

TensorRT engines are not portable across arbitrary TensorRT versions and GPU
architectures, so the container tag and GPU model are part of the benchmark
configuration rather than incidental setup details.

## Build the FP16 engine

The build configuration is intentionally bounded to batch size 1 and a
256-token maximum sequence length. This covers the controlled 28-input plus
128-output-token test. A later Triton serving engine will use a larger shape
envelope suitable for concurrent requests.

```bash
cd /workspace/TritonLLMDeployment
source scripts/runpod_env.sh

python phase2/build_fp16_engine.py \
  | tee outputs/trtllm-fp16-build.txt
```

The saved engine is written to `artifacts/trtllm/fp16-engine` and is ignored
by Git because it is large and specific to this software/hardware stack.

## Benchmark the FP16 engine

```bash
python phase2/benchmark_engine.py \
  --engine-dir artifacts/trtllm/fp16-engine \
  --warmup-runs 3 \
  --runs 10 \
  --max-new-tokens 128 \
  | tee outputs/trtllm-fp16-rtx4090-batch1.txt
```

The benchmark supplies token IDs directly to TensorRT-LLM. Tokenization and
detokenization therefore remain outside the timed region, matching Phase 1.
TensorRT-LLM preallocates a KV-cache pool, so the script limits that pool to
10% of otherwise-free device memory for this small batch-one test and labels
its memory result as total device memory in use rather than PyTorch-allocated
memory.

## FP8 follows FP16

Do not build FP8 until the FP16 engine is verified. FP8 requires calibration:
representative text is passed through the FP16 model to estimate scaling
ranges that map higher-precision weights and activations into FP8. Calibration
data affects quality, so its dataset and sample count will be selected and
documented before quantization.
