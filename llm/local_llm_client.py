import requests
import json
import time
from typing import Optional
from config import OLLAMA_MODEL
class LocalLLMClient:
    """
    Client for local LLM inference via Ollama.
    Models: qwen2:7b-instruct-q4_0 (or similar)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.model_name = OLLAMA_MODEL
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
    
    def generate_text_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7):
        """
        Generate text using local model with streaming.
        Yields chunks of text as they arrive.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=300, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    token = chunk.get("response", "")
                    if token:
                        full_response += token
                        yield token
                    if chunk.get("done", False):
                        break
                        
        except requests.exceptions.ConnectionError:
            yield f"[ERROR] Could not connect to Ollama at {self.base_url}. Is it running? (ollama serve)"
        except Exception as e:
            yield f"[ERROR] LLM inference failed: {str(e)}"


if __name__ == "__main__":
    client = LocalLLMClient()
    response = client.generate_text("What is the meaning of life?")
    print(response)
