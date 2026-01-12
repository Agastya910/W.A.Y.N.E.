import requests
import json
import time
from typing import Optional

class LocalLLMClient:
    """
    Client for local LLM inference via Ollama.
    Models: qwen2:7b-instruct-q4_0 (or similar)
    """
    
    def __init__(self, model_name: str = "qwen2:7b-instruct-q4_0", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/generate"
    
    def generate_text(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate text using local model.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return f"[ERROR] Could not connect to Ollama at {self.base_url}. Is it running? (ollama serve)"
        except Exception as e:
            return f"[ERROR] LLM inference failed: {str(e)}"


if __name__ == "__main__":
    client = LocalLLMClient()
    response = client.generate_text("What is the meaning of life?")
    print(response)
