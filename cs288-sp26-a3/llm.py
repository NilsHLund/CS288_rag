import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALLOWED_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "qwen/qwen3-8b",
    "qwen/qwen-2.5-7b-instruct",
    "allenai/olmo-3-7b-instruct",
    "mistralai/mistral-7b-instruct"
    ]
DEFAULT_MODEL = ALLOWED_MODELS[0]


def call_llm(
    query: str,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """
    Call OpenRouter chat completions and return the assistant text.

    Constraints:
    - Uses OPENROUTER_API_KEY from environment.
    - Allows only models in ALLOWED_MODELS.
    """
    assert model in ALLOWED_MODELS, (
        f"Model '{model}' is not allowed. Allowed models: {ALLOWED_MODELS}"
    )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"OpenRouter response missing choices: {data}")
