"""
OpenRouter LLM Provider
======================
Client for the OpenRouter API. Supports both blocking and real-time
streaming (SSE) to avoid 429 rate-limit errors on large responses.
"""

import json
import os
import urllib.request
import requests as req_lib
from typing import Optional, Dict, Iterator, List, Tuple
from pathlib import Path
from functools import lru_cache
from threading import Lock

from dune.llm.discovery_prompt import discovery_format_response


class OpenRouterClient:
    MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
    DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    # Preferred free models in priority order
    PREFERRED_FREE_MODELS = [
        "google/gemma-4-26b-a4b-it:free",
        "meta-llama/llama-3-3-70b-instruct:free",
        "venice/uncensored:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "qwen/qwen3-coder-480b-a35b-instruct:free",
        "meta-llama/llama-3-2-3b-instruct:free",
        "nousresearch/hermes-3-405b-instruct:free",
        "openrouter/free",
        "liquidai/lfm2-5-1-2b-thinking:free",
        "liquidai/lfm2-5-1-2b-instruct:free",
        "nvidia/nemotron-3-5-content-safety:free",
        "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
    ]

    def __init__(self, api_key: str = "", model: str = ""):
        # Load API key: explicit arg > OPENROUTER_API_KEY env var > default
        env_key = os.environ.get('OPENROUTER_API_KEY', '')
        self.api_key = api_key or env_key
        # Load model: explicit arg > OPENROUTER_MODEL env var > first free model
        env_model = os.environ.get('OPENROUTER_MODEL', '')
        self.model = model or env_model
        if not self.model:
            self.model = self.PREFERRED_FREE_MODELS[0] if self.PREFERRED_FREE_MODELS else ""
        self.endpoint = self.DEFAULT_ENDPOINT
        self._models_cache = None
        # Manual model selection only - no auto-selection
        self._discovery_template_path: Optional[Path] = None
        # Connection pooling: reuse HTTP sessions for keep-alive
        self._session = req_lib.Session()
        self._session.headers.update({
            "HTTP-Referer": "http://localhost:8081",
            "X-Title": "DUNE",
        })
        # Cache lock for thread safety
        self._cache_lock = Lock()
        # In-memory response cache: (model, message_json) -> response
        self._response_cache: Dict[Tuple[str, str], str] = {}
        self._cache_max_size = 256

    def _get_cached(self, model: str, messages: list) -> str | None:
        """Get cached response if available."""
        key = (model, json.dumps(messages, sort_keys=True))
        return self._response_cache.get(key)

    def _set_cache(self, model: str, messages: list, response: str) -> None:
        """Cache a response, evicting oldest if full."""
        key = (model, json.dumps(messages, sort_keys=True))
        with self._cache_lock:
            if len(self._response_cache) >= self._cache_max_size:
                # Evict first item (oldest)
                self._response_cache.pop(next(iter(self._response_cache)))
            self._response_cache[key] = response

    def fetch_all_models(self):
        """Fetch all models from OpenRouter API. Returns list of {id, name}."""
        if self._models_cache is not None:
            return self._models_cache
        try:
            req = urllib.request.Request(self.MODELS_ENDPOINT)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                models = data.get("data", [])
                all_models = [{"id": m["id"], "name": m.get("name", m["id"])}
                        for m in models]
                all_models.sort(key=lambda m: m["name"])
                self._models_cache = all_models
                return all_models
        except Exception:
            self._models_cache = []
            return []
    
    def _call(self, messages: list, max_tokens: int = 4096) -> str:
        """Blocking call — returns the full response string. Uses connection pooling."""
        if not self.api_key:
            return ""
        
        # Check cache first
        cached = self._get_cached(self.model, messages)
        if cached is not None:
            return cached
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        try:
            resp = self._session.post(
                self.endpoint,
                headers=headers,
                json=data,
                timeout=120,
            )
            resp.raise_for_status()
            res = resp.json()
            content = res['choices'][0]['message']['content']
            # Cache the response (skip errors)
            if not content.startswith("[LLM Error"):
                self._set_cache(self.model, messages, content)
            return content
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def _call_stream(self, messages: list, max_tokens: int = 4096) -> Iterator[str]:
        """
        Streaming call — yields tokens one at a time as they arrive
        from OpenRouter's SSE endpoint. This prevents 429 errors by
        consuming the response incrementally and keeping the connection
        alive rather than waiting for the entire response.
        """
        if not self.api_key:
            yield ""
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8081",
            "X-Title": "DUNE",
            "Accept": "text/event-stream",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            with self._session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                buffer = ""
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    buffer += chunk
                    # SSE: each event is separated by \n\n
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.split("\n"):
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    return
                                try:
                                    event_data = json.loads(data_str)
                                    delta = (
                                        event_data.get("choices", [{}])[0]
                                        .get("delta", {})
                                        .get("content", "")
                                    )
                                    if delta:
                                        yield delta
                                except json.JSONDecodeError:
                                    continue
        except Exception as e:
            yield f" [Stream Error: {str(e)}]"

    def extract_query(self, user_input: str) -> str:
        """Extract structured JSON from user input.
        
        Optimized: uses small max_tokens (128) since the output is tiny JSON.
        Falls back to wrapping input as object if LLM errors.
        """
        if not self.api_key:
            return f'{{"object": "{user_input}"}}'
        messages = [
            {"role": "system", "content": "You are the Input LLM for the DUNE cognitive engine. Your ONLY job is to convert messy human language into a structured JSON concept. Do not reason. Do not add conversational text. Output ONLY valid JSON in this exact format: {\"domain\": \"...\", \"object\": \"...\", \"goal\": \"...\", \"environment\": \"...\"}"},
            {"role": "user", "content": user_input}
        ]
        q = self._call(messages, max_tokens=128).strip()
        if not q or q.startswith("[LLM Error"):
            return f'{{"object": "{user_input}"}}'
        return q

    def format_response(self, user_input: str, dune_trace: str) -> str:
        """Non-streaming response (complete string)."""
        if not self.api_key:
            return "Error: API key is required to generate text response."

        messages = discovery_format_response(
            user_query=user_input,
            context=dune_trace,
            template_path=self._discovery_template_path,
        )
        ans = self._call(messages, max_tokens=4096)
        if ans and not ans.startswith("[LLM Error"):
            return ans
        return ans

    def format_response_stream(
        self, user_input: str, dune_trace: str
    ) -> Iterator[str]:
        """
        Streaming response — yields tokens in real-time via SSE.
        
        This is the 429-safe path: the server consumes tokens one by one
        instead of waiting for the full response, keeping the HTTP connection
        active and avoiding rate-limit backpressure.
        """
        if not self.api_key:
            yield "Error: API key is required to generate text response."
            return

        messages = discovery_format_response(
            user_query=user_input,
            context=dune_trace,
            template_path=self._discovery_template_path,
        )

        yield from self._call_stream(messages, max_tokens=4096)

    def get_status(self) -> dict:
        """Get client status overview."""
        return {
            'provider': 'OpenRouter',
            'available': bool(self.api_key),
            'current_model': self.model,
            'api_configured': bool(self.api_key),
        }
