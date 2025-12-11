from __future__ import annotations

"""Qwen2.5-VL vision-language model adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .config import load_settings


@dataclass
class QwenVLSettings:
    model_id: str
    device: str
    max_new_tokens: int
    temperature: float


def _load_qwen_settings() -> QwenVLSettings:
    """Extract Qwen2.5-VL settings from the shared config."""
    s = load_settings()
    return QwenVLSettings(
        model_id=s.vlm_model_id,
        device=s.vlm_device,
        max_new_tokens=s.vlm_max_new_tokens,
        temperature=s.vlm_temperature,
    )


class QwenVisionModel:
    """Small wrapper around Qwen2.5-VL for image + text generation."""

    def __init__(self) -> None:
        self.settings = _load_qwen_settings()
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.settings.model_id,
            device_map="auto" if self.settings.device == "cuda" else None,
        )

    def generate_from_image(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run a single-image + text generation call."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")

        max_tokens = max_new_tokens or self.settings.max_new_tokens
        temp = temperature if temperature is not None else self.settings.temperature

        # Qwen2.5-VL expects a chat-style prompt with an image placeholder,
        # built via `apply_chat_template`. This ensures that image features
        # and image tokens are aligned.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[chat_prompt],
            images=[image],
            return_tensors="pt",
        )

        if self.settings.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=temp > 0,
        )

        # For many Qwen-VL checkpoints, the processor handles decoding.
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return output.strip()


_qwen_vision_model: Optional[QwenVisionModel] = None


def get_qwen_vision_model() -> QwenVisionModel:
    """Lazy-load and return the singleton Qwen2.5-VL model instance."""
    global _qwen_vision_model
    if _qwen_vision_model is None:
        _qwen_vision_model = QwenVisionModel()
    return _qwen_vision_model
