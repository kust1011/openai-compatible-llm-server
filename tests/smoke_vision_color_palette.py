"""
Simple smoke test for the Qwen2.5-VL vision endpoint.

Run this after starting the server with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests


def build_url() -> str:
    """Build the vision color palette URL from environment variables."""
    port = int(os.getenv("PORT", "8000"))
    return f"http://127.0.0.1:{port}/v1/vision/color_palette"


def make_test_payload() -> Dict[str, Any]:
    """Create a minimal, valid vision color analysis payload."""
    # Default to the dog image under tests/, but allow override via env.
    root = Path(__file__).resolve().parents[1]
    default_image = root / "tests" / "dog.jpeg"
    image_path = os.getenv("VLM_TEST_IMAGE", str(default_image))

    prompt = (
        "You are a concise vision-language assistant. "
        "Describe the main colors in the image in a short JSON object with "
        "keys: colors (list of strings) and notes (short string). "
        "Return only JSON."
    )

    return {
        "image_path": image_path,
        "prompt": prompt,
        "temperature": float(os.getenv("VLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("VLM_MAX_NEW_TOKENS", "256")),
    }


def main() -> int:
    url = build_url()
    payload = make_test_payload()

    print(f"Sending vision test request to: {url}")
    print(f"Using image: {payload['image_path']}")

    if not Path(payload["image_path"]).exists():
        print(f"[ERROR] Image does not exist: {payload['image_path']}", file=sys.stderr)
        return 1

    try:
        response = requests.post(url, json=payload, timeout=120)
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

    result = data.get("result", "")
    if not isinstance(result, str) or not result.strip():
        print("[ERROR] Empty vision result.", file=sys.stderr)
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    print("Vision smoke test succeeded. Raw model result:")
    print(result.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

