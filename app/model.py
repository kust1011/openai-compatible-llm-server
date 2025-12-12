from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import load_settings


class LocalChatModel:
    """Wrapper around a HuggingFace causal LM for simple chat-style generation."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
        device = self.settings.device

        # When device is exactly "cuda", let accelerate infer a sharded
        # device_map across all visible GPUs. For any other value
        # (e.g. "cuda:0", "cuda:1", "cpu", "mps"), load on that device only.
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_id,
                device_map="auto",
                torch_dtype="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_id,
                torch_dtype="auto",
            )
            # This supports explicit devices like "cuda:0", "cuda:1", "cpu", "mps".
            self.model.to(device)

        # When using accelerate with device_map="auto", the model is already
        # placed on the appropriate devices and the pipeline must not receive
        # an explicit device argument. For single-device placement, the model's
        # own device is respected by the pipeline.
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(self, messages: List[dict], max_new_tokens: int | None = None, temperature: float | None = None) -> str:
        """
        Very simple chat formatting: concatenate messages into a single prompt.

        You can replace this with proper chat templates if your model provides one.
        """
        max_tokens = max_new_tokens or self.settings.max_new_tokens
        temp = temperature if temperature is not None else self.settings.temperature

        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        prompt_parts.append("ASSISTANT:")
        prompt = "\n".join(prompt_parts)

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=temp > 0,
        )

        text = outputs[0]["generated_text"]
        # Return only the part after the assistant marker.
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", maxsplit=1)[-1].strip()
        return text.strip()


local_model: LocalChatModel | None = None


def get_model() -> LocalChatModel:
    """Lazy-load and return the singleton local model instance."""
    global local_model
    if local_model is None:
        local_model = LocalChatModel()
    return local_model
