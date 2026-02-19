import ollama
from config import OLLAMA_MODEL


class LocalLLMClient:
    """
    Client for local LLM inference via Ollama.
    Uses the official ollama Python library.
    """

    def __init__(self):
        self.model_name = OLLAMA_MODEL
        self.client = ollama.Client()

    def generate_text(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate text using local model. Returns full response string."""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            return response["response"].strip()
        except Exception as e:
            return f"[ERROR] LLM inference failed: {str(e)}"

    def generate_text_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Generate text with streaming. Yields text chunks."""
        try:
            stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            for chunk in stream:
                token = chunk.get("response", "")
                if token:
                    yield token
        except Exception as e:
            yield f"[ERROR] LLM inference failed: {str(e)}"

    def chat(self, messages: list, temperature: float = 0.7, json_mode: bool = False) -> str:
        """
        Chat-style generation. Optionally enforces JSON output.
        messages: list of {"role": "...", "content": "..."} dicts
        """
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "options": {"temperature": temperature}
            }
            if json_mode:
                kwargs["format"] = "json"
            response = self.client.chat(**kwargs)
            return response["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR] Chat inference failed: {str(e)}"
