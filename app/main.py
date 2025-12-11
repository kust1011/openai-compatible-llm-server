from typing import List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .model import get_model


app = FastAPI(title="Agentic Outfit LLM Server")


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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completion endpoint backed by a local model.

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


@app.get("/health")
def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}

