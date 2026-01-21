from typing import List, Optional
from openai import OpenAI
from base_llm import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible LLM wrapper. Works with OpenAI, vLLM, and other compatible APIs."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        """
        Initialize the OpenAI-compatible LLM.

        Args:
            model: Model name/identifier.
            base_url: Base URL for the API (use for vLLM or other compatible servers).
            api_key: API key (uses OPENAI_API_KEY env var if not provided).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response using retrieved context."""
        default_system = (
            "You are a helpful assistant. Answer the user's question based on the "
            "provided context. If the context doesn't contain relevant information, "
            "say so clearly."
        )

        context_text = "\n\n".join(context)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt or default_system,
        )
