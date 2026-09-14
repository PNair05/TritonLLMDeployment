"""Small standard-library client for Triton's generate extension."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def format_mistral_instruction(message: str) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("The prompt cannot be empty")
    return f"[INST] {cleaned} [/INST]"


class TritonClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 100.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict | str:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Triton returned HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Triton at {self.base_url}: {exc.reason}") from exc

        if not body:
            return ""
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    def server_ready(self) -> bool:
        try:
            self._request("/v2/health/ready")
            self._request("/v2/models/ensemble/ready")
        except RuntimeError:
            return False
        return True

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        if not 1 <= max_tokens <= 128:
            raise ValueError("max_tokens must be between 1 and 128 for the current engine")
        response = self._request(
            "/v2/models/ensemble/generate",
            {
                "text_input": format_mistral_instruction(prompt),
                "max_tokens": max_tokens,
                "bad_words": "",
                "stop_words": "",
                "exclude_input_in_output": True,
                "top_k": 1,
            },
        )
        if not isinstance(response, dict) or "text_output" not in response:
            raise RuntimeError(f"Unexpected Triton response: {response!r}")
        output = response["text_output"]
        if isinstance(output, list):
            output = output[0]
        if not isinstance(output, str):
            raise RuntimeError(f"Unexpected text_output value: {output!r}")
        return output.strip()
