# Project: LLM Deployment Optimization with TensorRT-LLM + Triton Inference Server

## Who I am / context for you (the assistant)
I'm a CS undergrad (rising junior) with experience in:
- **C++ systems programming** — built a multithreaded Redis clone from scratch (POSIX sockets, TCP/IP, mutex-based concurrency, custom RESP parser, STL hash tables)
- **LLM application development** — built an agentic AI platform (LangChain + Next.js + Python) and a voice-driven LLM assistant with a custom safety/validation layer
- **Cloud infrastructure** — deployed and operated a production REST API on AWS EC2 with PM2, zero-downtime deployments
- **No prior CUDA/GPU programming or ML deployment experience** — this is new territory

I'm doing this project to build real, defensible knowledge of NVIDIA's inference stack ahead of a recruiting event, targeting a Solutions Architect Intern role. I do NOT have a hard deadline to finish — I have about 4 weeks before the event, but the project can still be in progress after that.

## Goal
Take a small open-weight LLM, optimize it for inference using **TensorRT-LLM**, serve it through **NVIDIA Triton Inference Server**, and wrap it with a minimal application (chat or RAG interface) that actually uses the deployed endpoint. Benchmark optimized vs. baseline inference and document the results.

## How I want you to help me (important — read this before generating anything)
This project only has value to me if I understand every decision in it well enough to explain it out loud to an engineer, with no notes. So:

- **Teach before you generate.** When we hit a new concept (quantization modes, KV-cache, calibration datasets, Triton's model repository format, etc.), explain it to me first in plain language. Don't just hand me a working script.
- **It's fine to generate boilerplate and fix environment/dependency errors directly** — Dockerfiles, CUDA/driver version wrangling, config scaffolding, stack-trace debugging. That's friction, not learning, and I don't need to suffer through it manually.
- **For the core technical decisions** (which quantization precision to use, how to structure the benchmark, what the calibration set should look like), give me the options and tradeoffs, and let me make the call — then sanity-check my reasoning rather than overriding it.
- **Check my understanding periodically.** Occasionally ask me to explain back a piece we just built, especially before we move to the next phase.
- Assume I have access to a cloud GPU instance (Colab Pro, RunPod, Lambda Labs, or AWS `g5`) — help me pick the simplest option for each phase rather than defaulting to the most powerful/expensive one.

## Planned phases

**Phase 1 — Baseline**
Get a small open-weight model (leaning toward Llama 3 8B or Mistral 7B; open to Phi-3-mini if compute-constrained) running with standard HuggingFace/PyTorch inference on a cloud GPU. Measure baseline latency/throughput. No optimization yet — just a working, timed baseline.

**Phase 2 — TensorRT-LLM optimization**
Quantize the model (INT8 or FP8) using TensorRT-LLM. Re-benchmark against the Phase 1 baseline. Document the real speedup number and what changed.

**Phase 3 — Serve + wrap**
Stand the optimized model up behind Triton Inference Server with a real endpoint. Build a minimal chat or RAG interface on top of it (reusing patterns from my past LLM projects). This phase is fine to leave partially finished — a clear README explaining what's done and what's next is a legitimate deliverable on its own.

## Fallback plan
If TensorRT-LLM setup burns too much time without a working benchmark, fall back to plain TensorRT or ONNX Runtime's CUDA execution provider on a smaller model rather than staying stuck. A working, explainable result on a simpler tool beats a broken attempt at a fancier one.

## Deliverables I actually care about
1. A real, honest benchmark number (e.g., "Nx faster after quantization" at a stated batch size/sequence length)
2. A README that explains *why* I made the decisions I made, not just what the result was
3. Clean, honest git history (small commits, real commit messages) — not one giant commit that appears finished
4. The ability to explain the whole pipeline, including the parts that didn't go smoothly

---
Let's start with Phase 1. Ask me what cloud GPU environment I want to use, then walk me through getting a baseline model loaded and a timed inference run working before we touch TensorRT-LLM at all.
