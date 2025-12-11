## Agentic Outfit LLM Server

CLI-only backend server exposing an OpenAI-compatible chat completion API, designed to run a local LLM (e.g. HuggingFace model) and serve the Agentic Outfit Stylist frontend.

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

   The Agentic Outfit Stylist repo can then talk to this server using:

   - `OPENAI_BASE_URL=http://localhost:8000/v1`
   - `OPENAI_API_KEY` set to any non-empty string.

### Notes

- The default implementation uses HuggingFace `transformers`. You can swap in vLLM or another backend inside `app/model.py`.

