import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    model_id: str
    device: str
    max_new_tokens: int
    temperature: float
    host: str
    port: int
    vlm_model_id: str
    vlm_device: str
    vlm_max_new_tokens: int
    vlm_temperature: float


def load_settings() -> Settings:
    """Load settings from environment variables or .env."""
    load_dotenv()

    # Normalize Hugging Face token environment variable for gated models.
    # Transformers and huggingface_hub look for HF_TOKEN or HUGGINGFACE_HUB_TOKEN.
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("hf_token")
    )
    if hf_token and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    return Settings(
        model_id=os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device=os.getenv("DEVICE", "cuda"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        vlm_model_id=os.getenv(
            "VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"
        ),
        vlm_device=os.getenv("VLM_DEVICE", os.getenv("DEVICE", "cuda")),
        vlm_max_new_tokens=int(os.getenv("VLM_MAX_NEW_TOKENS", "512")),
        vlm_temperature=float(os.getenv("VLM_TEMPERATURE", "0.2")),
    )
