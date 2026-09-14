# Phase 3: Triton serving and application wrapper

This phase puts the validated TensorRT-LLM engine behind Triton and calls its
HTTP endpoint from a minimal terminal chat application.

## What each layer does

- `preprocessing` runs the Hugging Face tokenizer and turns text into token IDs.
- `tensorrt_llm` schedules requests and executes the compiled engine on the GPU.
- `postprocessing` turns output token IDs back into text.
- `ensemble` exposes those three steps as one Triton model and one HTTP request.
- `chat.py` formats user messages for Mistral and calls Triton's generate API.

The first serving pass deliberately reuses the Phase 2 FP16 engine. Its build
limits are batch size 1, 128 input tokens, and 256 total tokens, so this is a
single-user, short-prompt functional deployment rather than the final
concurrency configuration.

## Run on the existing RunPod

Use the same `nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3` container and
the same RTX 4090 where the engine was built. From the project directory:

```bash
cd /workspace/TritonLLMDeployment
git pull
source scripts/runpod_env.sh

python phase3/prepare_model_repository.py
./phase3/serve.sh
```

The preparation script copies the model templates bundled with the pinned
container, fills them with the engine and tokenizer paths, and writes the
generated repository to `artifacts/triton/model-repository`. That directory is
machine-specific and intentionally ignored by Git.

Wait until Triton logs show that HTTP is listening on port 8000 and all four
models are `READY`. Keep that terminal running. In a second RunPod terminal:

```bash
cd /workspace/TritonLLMDeployment
python phase3/smoke_test.py
python phase3/chat.py
```

The smoke test must report both the server and ensemble model as ready, then
print a generated response.

## Access from outside the pod

The safest development option is an SSH tunnel that maps your computer's port
8000 to the pod's port 8000; the client can then keep its default URL. If using
RunPod's HTTP proxy instead, expose HTTP port `8000` in the pod settings and use:

```bash
python phase3/chat.py \
  --url https://POD_ID-8000.proxy.runpod.net
```

RunPod proxy URLs are publicly reachable. This prototype has no authentication
or rate limiting, so do not leave port 8000 exposed after testing.

## Next serving iteration

After the endpoint is verified, build a separate serving engine with a larger
input/sequence envelope and batch size greater than one. Then add concurrent
load tests and compare direct TensorRT-LLM latency with Triton end-to-end
latency. Keep the benchmark engine unchanged so the Phase 2 numbers remain
reproducible.
