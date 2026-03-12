import ollama
from app.config import settings


def generate_with_biomistral(prompt: str) -> str:
    response = ollama.generate(
        model=settings.BIOMISTRAL_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.2,
            "top_p": 0.9,
        }
    )
    return response["response"]