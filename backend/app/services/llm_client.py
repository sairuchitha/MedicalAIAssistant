import ollama
from app.config import settings

_client = ollama.Client(timeout=300)


def generate_with_llm(prompt: str) -> str:
    response = _client.generate(
        model=settings.OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": 0,
            "top_p": 0.9,
        },
    )
    return response["response"]