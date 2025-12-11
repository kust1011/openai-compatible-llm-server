from __future__ import annotations

"""
Qwen2.5-VL vision-language model adapter.

This module provides a small wrapper around a local Qwen2.5-VL-7B-Instruct
checkpoint for single-image + text generation. It is wired into the API
as a generic vision completion endpoint so other projects can reuse it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
    """
    Wrapper around Qwen2.5-VL-7B-Instruct for simple vision + text generation.

    This implementation assumes the checkpoint is compatible with
    `AutoProcessor` and `AutoModelForCausalLM` from `transformers`. If you use
    a different Qwen variant, adjust the model class or preprocessing here.
    """

    def __init__(self) -> None:
        self.settings = _load_qwen_settings()
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
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
        """
        Run a single-image + text generation with Qwen2.5-VL.

        The prompt should instruct the model to return a compact, parseable
        answer (e.g., JSON for color analysis).
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")

        max_tokens = max_new_tokens or self.settings.max_new_tokens
        temp = temperature if temperature is not None else self.settings.temperature

        inputs = self.processor(
            text=[prompt],
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

