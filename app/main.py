from typing import List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .model import get_model
from .vlm_qwen import get_qwen_vision_model


app = FastAPI(title="OpenAI-Compatible Local LLM Server")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: List[ChatCompletionChoice]


class VisionColorAnalysisRequest(BaseModel):
    image_path: str
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class VisionColorAnalysisResponse(BaseModel):
    result: str


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completion endpoint backed by a local text model.

    This is a minimal compatibility layer, sufficient for use with
    `langchain-openai`'s `ChatOpenAI` client.
    """
    model = get_model()
    output = model.generate(
        [m.model_dump() for m in req.messages],
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    message = ChatMessage(role="assistant", content=output)

    return ChatCompletionResponse(
        id="chatcmpl-local-1",
        object="chat.completion",
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason="stop",
            )
        ],
    )


@app.post(
    "/v1/vision/color_palette",
    response_model=VisionColorAnalysisResponse,
    tags=["vision"],
)
def vision_color_palette(req: VisionColorAnalysisRequest) -> VisionColorAnalysisResponse:
    """
    Run a generic vision-language generation task using Qwen2.5-VL.

    Callers should provide a prompt that clearly specifies the task and
    desired output format. If no prompt is provided, the model is asked to
    briefly describe the main contents of the image.
    """
    default_prompt = (
        "You are a concise vision-language assistant. "
        "Describe the main contents of the image in a few sentences."
    )
    prompt = req.prompt or default_prompt

    model = get_qwen_vision_model()
    result = model.generate_from_image(
        image_path=req.image_path,
        prompt=prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    return VisionColorAnalysisResponse(result=result)


@app.get("/health")
def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}
