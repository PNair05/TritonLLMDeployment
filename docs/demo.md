# Verified Triton demo

The following terminal output was captured from the RunPod RTX 4090 deployment
on September 13, 2026. It demonstrates application behavior; benchmark numbers
are documented separately in [`results/`](../results/README.md).

## Readiness and inference smoke test

```text
$ python phase3/smoke_test.py
Triton server: ready
Ensemble model: ready

Response
An Inference Server is a software component that executes machine learning
models and makes predictions or inferences based on input data, facilitating
the integration of AI capabilities into applications and services. It
optimizes the deployment of models, handling tasks such as scaling, monitoring,
and managing resources, to ensure efficient and accurate inference
```

Before the script was run, the underlying checks also returned HTTP 200:

```text
GET /v2/health/ready            -> HTTP 200
GET /v2/models/ensemble/ready   -> HTTP 200
```

Together, these checks show that the server process was reachable, the
ensemble and its dependencies loaded successfully, and an end-to-end request
could pass through preprocessing, TensorRT-LLM execution, and postprocessing.

## Interactive client

```text
$ python phase3/chat.py
Connected to Mistral 7B through Triton.
This first version is single-turn because the engine has a 128-token input limit.
Type quit to exit.

you> explain the history of the flatiron building in 4-5 sentences.
model> The Flatiron Building, located in Manhattan, New York, was constructed between 1902 and 1903. Designed by Daniel Burnham and Dankmar Adler, with the distinctive triangular footprint created by the intersection of Fifth Avenue, Broadway, and 23rd Street, it was one of the first skyscrapers inthe city. The building, owned by Harry S. Black, was a symbol of modernity and

you> quit
```

This transcript came from the initial 96-output-token client default, so the
answer was visibly truncated. The default is now 128 output tokens, matching
the maximum output length used in the engine benchmark. The transcript is
retained unchanged rather than presenting generated content as a polished or
fact-checked answer.

## Run the demo

Start Triton in the first terminal:

```bash
cd /workspace/TritonLLMDeployment
source scripts/runpod_env.sh
python phase3/prepare_model_repository.py
./phase3/serve.sh
```

If the generated model repository already exists, skip the preparation command
and launch the server directly. In a second terminal:

```bash
cd /workspace/TritonLLMDeployment
python phase3/smoke_test.py
python phase3/chat.py
```

