## OpenAI-Compatible Local LLM Server

CLI-only backend server exposing an OpenAI-compatible chat completion API, designed to run a local LLM (e.g. HuggingFace model). It can serve as a drop-in backend for any OpenAI-style client, including the Agentic Outfit Stylist frontend.

### Features

- OpenAI-style `POST /v1/chat/completions` endpoint.
- Uses environment variables for model and runtime settings.
- Intended to wrap a local HuggingFace model (GPU or CPU).

### Directory structure

- `app/`
  - `config.py` – settings and environment handling.
  - `model.py` – model loading and text generation.
  - `main.py` – FastAPI app with OpenAI-like API.
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

### Notes

- The default implementation uses HuggingFace `transformers`. You can swap in vLLM or another backend inside `app/model.py`.
