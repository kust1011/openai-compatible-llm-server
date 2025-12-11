"""
Simple smoke test for the OpenAI-compatible local LLM server.

Run this after starting the server with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import requests


def build_url() -> str:
    """Build the chat completions URL from environment variables."""
    port = int(os.getenv("PORT", "8000"))
    # The server usually listens on 0.0.0.0, but clients should call localhost.
    return f"http://127.0.0.1:{port}/v1/chat/completions"


def make_test_payload() -> Dict[str, Any]:
    """Create a minimal, valid chat completion payload."""
    return {
        "model": os.getenv("MODEL_ID", "local-test-model"),
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with a very short greeting."},
        ],
        "temperature": 0.7,
        "max_tokens": 64,
    }


def main() -> int:
    url = build_url()
    payload = make_test_payload()

    print(f"Sending test request to: {url}")
    try:
        response = requests.post(url, json=payload, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to connect to server: {exc}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"[ERROR] Server returned status {response.status_code}", file=sys.stderr)
        print(response.text, file=sys.stderr)
        return 1

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Response is not valid JSON: {exc}", file=sys.stderr)
        print(response.text, file=sys.stderr)
        return 1

    try:
        choice = data["choices"][0]
        message = choice["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Unexpected response structure: {exc}", file=sys.stderr)
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    if not isinstance(message, str) or not message.strip():
        print("[ERROR] Model response content is empty.", file=sys.stderr)
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    print("Smoke test succeeded. Model response:")
    print(message.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

