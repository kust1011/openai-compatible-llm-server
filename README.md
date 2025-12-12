## OpenAI-Compatible Local LLM Server

CLI-only backend server exposing an OpenAI-compatible chat completion API, designed to run a local LLM (e.g. HuggingFace model). It can serve as a drop-in backend for any OpenAI-style client that speaks the OpenAI API.

### Features

- OpenAI-style `POST /v1/chat/completions` endpoint.
- Optional Qwen2.5-VL-7B-Instruct endpoint for vision-language tasks such as color analysis.
- Uses environment variables for model and runtime settings.
- Intended to wrap a local HuggingFace model (GPU or CPU).

### Directory structure

- `app/`
  - `config.py` – settings and environment handling.
  - `model.py` – text model loading and generation.
  - `vlm_qwen.py` – Qwen2.5-VL-7B-Instruct vision-language adapter.
  - `main.py` – FastAPI app exposing text and vision endpoints.
- `.env.example` – example runtime configuration.

### Setup

1. Create and activate a Python environment (conda or venv).

   Using conda:

   ```bash
   conda create -n openai-llm-server python=3.11
   conda activate openai-llm-server
   ```

2. Copy `.env.example` to `.env` and edit values:

   ```bash
   cp .env.example .env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the server (CLI only):

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   Any OpenAI-compatible client can talk to this server using:

   - `OPENAI_BASE_URL=http://localhost:8000/v1`
   - `OPENAI_API_KEY` set to any non-empty string.

### Device configuration

- `DEVICE` controls where the text LLM is loaded. Valid values include:
  - `cpu`, `cuda`, `cuda:0`, `cuda:1`, `mps`, etc.
- `VLM_DEVICE` controls where the vision-language model is loaded and can be
  set independently from `DEVICE`.
- When `DEVICE=cuda` (without an index), the text model uses
  `device_map="auto"` to let `accelerate` shard across all visible GPUs.
- When you specify an explicit device (for example `DEVICE=cuda:0` and
  `VLM_DEVICE=cuda:1`), each model is loaded fully on that device only.

### Smoke test

After the server is running, you can run a simple smoke test from another terminal:

```bash
python -m tests.smoke_chat_completion
```

This script sends a small chat completion request to `http://127.0.0.1:$PORT/v1/chat/completions` and checks that the response has a valid OpenAI-style structure and a non-empty assistant message.


### Vision endpoint (Qwen2.5-VL)

If you configure `VLM_MODEL_ID` to a Qwen2.5-VL checkpoint (for example `Qwen/Qwen2.5-VL-7B-Instruct`), the server exposes a simple vision-language endpoint:

```bash
POST /v1/vision/color_palette
```

Request body:

```json
{
  "image_path": "/abs/path/to/portrait.jpg",
  "prompt": "Optional custom prompt for the VLM",
  "temperature": 0.2,
  "max_tokens": 512
}
```

The response contains a single field:

```json
{
  "result": "<model-generated text>"
}
```

By default, if you omit `prompt`, the server sends a built-in instruction for personal color analysis and expects the model to return a compact JSON description of the user's color season and recommended palettes.

### Notes

- The default implementation uses HuggingFace `transformers`. You can swap in vLLM or another backend inside `app/model.py`.
