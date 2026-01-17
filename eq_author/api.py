"""OpenAI API client and chat operations."""

import json
import hashlib
from typing import List, Dict, Any, Optional, Callable


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class PromptCache:
    """File-backed cache for chat responses to avoid repeat prompting."""

    def __init__(self, cache_dir) -> None:
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, *, model: str, messages: List[Dict[str, str]], temperature: Optional[float]):
        from pathlib import Path
        payload = {"model": model, "messages": messages, "temperature": temperature}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, *, model: str, messages: List[Dict[str, str]], temperature: Optional[float]) -> Optional[str]:
        cache_path = self._cache_path(model=model, messages=messages, temperature=temperature)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data.get("content")

    def set(self, *, model: str, messages: List[Dict[str, str]], temperature: Optional[float], content: str) -> None:
        cache_path = self._cache_path(model=model, messages=messages, temperature=temperature)
        cache_path.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")


def make_client(api_key: str, base_url: str) -> Any:
    """Create and test OpenAI client connection."""
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install with: pip install openai")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"Testing connection to: {base_url}")
        models = client.models.list()
        print(f"Connection successful. Available models: {len(models.data)}")
        return client
    except Exception as e:
        error_msg = f"Failed to connect to API at {base_url}: {str(e)}"
        if "Connection" in str(e) or "connect" in str(e).lower():
            error_msg += "\n\nConnection troubleshooting tips:"
            error_msg += "\n- Ensure the server is running and accessible"
            error_msg += "\n- Check if the URL is correct (e.g., http://localhost:1234 for LM Studio)"
            error_msg += "\n- Verify no firewall is blocking the connection"
            error_msg += "\n- For LM Studio, make sure 'Server' mode is enabled"
        elif "401" in str(e) or "unauthorized" in str(e).lower():
            error_msg += "\n\nAuthentication troubleshooting tips:"
            error_msg += "\n- Check if API key is correct"
            error_msg += "\n- For LM Studio, API key may not be required (use any non-empty string)"
        elif "404" in str(e) or "not found" in str(e).lower():
            error_msg += "\n\nEndpoint troubleshooting tips:"
            error_msg += "\n- Verify the base URL is correct"
            error_msg += "\n- Check if the API endpoint path is correct"
        raise RuntimeError(error_msg)


def chat_once(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    temperature: Optional[float] = None,
    on_token: Optional[Callable[[str], None]] = None,
    cache: Optional[PromptCache] = None,
    max_tokens: Optional[int] = None,
    get_default_output_tokens_func: Optional[Callable] = None,
) -> str:
    """Execute a single chat completion with caching support."""
    kwargs: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        if get_default_output_tokens_func is not None:
            kwargs["max_tokens"] = get_default_output_tokens_func(model)
        else:
            kwargs["max_tokens"] = max_tokens

    if cache is not None:
        cached = cache.get(model=model, messages=messages, temperature=temperature)
        if cached is not None:
            if stream and on_token is not None:
                try:
                    on_token(cached)
                except Exception:
                    pass
            return cached

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        error_msg = f"API request failed: {str(e)}"
        if "timeout" in str(e).lower():
            error_msg += "\n\nTimeout troubleshooting tips:"
            error_msg += "\n- The request took too long to complete"
            error_msg += "\n- Try again with a shorter prompt or simpler request"
            error_msg += "\n- Check if the server is overloaded"
        elif "rate" in str(e).lower() and "limit" in str(e).lower():
            error_msg += "\n\nRate limit troubleshooting tips:"
            error_msg += "\n- Wait a moment before making another request"
            error_msg += "\n- For LM Studio, this shouldn't occur locally"
        elif "model" in str(e).lower() and "not" in str(e).lower() and "found" in str(e).lower():
            error_msg += "\n\nModel troubleshooting tips:"
            error_msg += "\n- Check if the model name is correct"
            error_msg += "\n- For LM Studio, ensure a model is loaded"
            error_msg += "\n- Use client.models.list() to see available models"
        raise RuntimeError(error_msg)

    if stream:
        full = []
        try:
            for chunk in resp:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        text = delta.content
                        full.append(text)
                        if on_token is not None:
                            try:
                                on_token(text)
                            except Exception:
                                pass
        except Exception as e:
            error_msg = f"Streaming error occurred: {str(e)}"
            error_msg += "\n\nStreaming troubleshooting tips:"
            error_msg += "\n- Connection may have been interrupted"
            error_msg += "\n- Try again with streaming disabled (--no-stream)"
            raise RuntimeError(error_msg)
        result = "".join(full)
    else:
        try:
            result = resp.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            error_msg = f"Invalid response format from API: {str(e)}"
            error_msg += "\n\nResponse format troubleshooting tips:"
            error_msg += "\n- The API response structure was unexpected"
            error_msg += "\n- Check if the server is functioning correctly"
            error_msg += "\n- Try with a different model or server"
            raise RuntimeError(error_msg)

    if cache is not None:
        try:
            cache.set(model=model, messages=messages, temperature=temperature, content=result)
        except Exception:
            pass

    return result
