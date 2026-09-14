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

The RunPod used for this experiment had a 41,000,000,000-byte cgroup RAM limit
(about 38.2 GiB). TensorRT reported a 39.4 GB build-phase peak. Loading the
Hugging Face model and building in one Python process exhausted that limit.
The build script therefore converts the cached model shard-by-shard on CPU,
ends the converter, and replaces itself with `trtllm-build`. This prevents the
Hugging Face model and TensorRT builder from occupying RAM simultaneously.

```bash
cd /workspace/TritonLLMDeployment
source scripts/runpod_env.sh

mkdir -p outputs
nohup python -u phase2/build_fp16_engine.py \
  > outputs/trtllm-fp16-build.txt 2>&1 &

tail -f outputs/trtllm-fp16-build.txt
```

The saved engine is written to `artifacts/trtllm/fp16-engine` and is ignored
by Git because it is large and specific to this software/hardware stack.

The intermediate checkpoint is written to
`artifacts/trtllm/fp16-checkpoint`. Both directories must contain their
respective `config.json` and rank-zero data file to be considered complete;
the script refuses to reuse a non-empty partial directory.

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

## FP16 benchmark result

The same-pod comparison on an RTX 4090 generated exactly 128 new tokens for
each measured request:

| Metric | PyTorch FP16 | TensorRT-LLM FP16 | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 3.924 s | 2.038 s | 48.1% lower |
| Median latency | 3.918 s | 2.037 s | 48.0% lower |
| Latency standard deviation | 0.026 s | 0.001 s | 96.2% lower |
| Generation throughput | 32.62 tokens/s | 62.82 tokens/s | 1.93x |

Engine loading took 40.28 seconds versus 26.37 seconds for the PyTorch model.
This is a startup tradeoff rather than a per-request cost for a persistent
service. The PyTorch and TensorRT-LLM GPU-memory numbers use different
measurement scopes and are therefore recorded but not compared directly.

## FP8 follows FP16

Do not build FP8 until the FP16 engine is verified. FP8 requires calibration:
representative text is passed through the FP16 model to estimate scaling
ranges that map higher-precision weights and activations into FP8. Calibration
data affects quality, so its dataset and sample count will be selected and
documented before quantization.
