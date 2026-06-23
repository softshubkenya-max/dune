import json
import urllib.request
from typing import Optional, Dict

class OpenRouterClient:
    MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or "OPENROUTER_API_KEY"
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self._models_cache = None
        # Auto-resolve model from API if none provided
        if not self.model:
            models = self.fetch_all_models()
            self.model = models[0]["id"] if models else "google/gemma-4-31b-it:free"

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
    
    def _call(self, messages: list) -> str:
        if not self.api_key:
            return ""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8081",
            "X-Title": "DUNE"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096
        }
        
        req = urllib.request.Request(self.endpoint, headers=headers, data=json.dumps(data).encode('utf-8'))
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                return res['choices'][0]['message']['content']
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def extract_query(self, user_input: str) -> str:
        if not self.api_key: return f'{{"object": "{user_input}"}}'
        messages = [
            {"role": "system", "content": "You are the Input LLM for the DUNE cognitive engine. Your ONLY job is to convert messy human language into a structured JSON concept. Do not reason. Do not add conversational text. Output ONLY valid JSON in this exact format: {\"domain\": \"...\", \"object\": \"...\", \"goal\": \"...\", \"environment\": \"...\"}"},
            {"role": "user", "content": user_input}
        ]
        q = self._call(messages).strip()
        if not q or q.startswith("[LLM Error"):
            return f'{{"object": "{user_input}"}}' # fallback
        return q

    def format_response(self, user_input: str, dune_trace: str) -> str:
        if not self.api_key: 
            return "Error: API key is required to generate text response."
            
        messages = [
            {"role": "system", "content": "You are a visionary, super-scientist conversational interface for a powerful cognitive engine named DUNE. DUNE has already performed the reasoning. Your job is to translate DUNE's raw reasoning trace into a fluent, highly in-depth, 'zero to hero' explanation. Break down the concepts with all details, explaining the fundamentals up to advanced visionary applications and new inventions, maintaining a brilliant super-scientist persona. IMPORTANT VISUALIZATION RULES: 1. When explaining structures, hierarchies, timelines, or processes, ALWAYS include a Mermaid.js diagram (using ```mermaid ... ``` code blocks). 2. When visualizing quantitative data, statistics, or mathematical relationships, provide D3.js code inside ```javascript ... ``` blocks. Assume `d3` (v7) is available. Your javascript code MUST append its SVG/chart to a DOM element with the exact ID 'd3-container' (e.g., `d3.select('#d3-container')`). 3. When tasked with drawing whiteboards, schematics, or hand-drawn conceptual diagrams, provide a JSON array of Excalidraw elements inside an ```excalidraw ... ``` block. Example: `[{\"type\": \"rectangle\", \"x\": 100, \"y\": 100, \"width\": 200, \"height\": 100, \"strokeColor\": \"#000\"}]`. Do NOT hallucinate core facts outside of what DUNE provides, but you may expand deeply on the scientific concepts for fluency and educational purposes."},
            {"role": "user", "content": f"User's original query: {user_input}\n\nDUNE's raw output:\n{dune_trace}\n\nPlease format this into an in-depth natural response."}
        ]
        
        # Try the selected model first
        ans = self._call(messages)
        if ans and not ans.startswith("[LLM Error"):
            return ans
            
        # If it failed (e.g. rate limit), auto-retry with up to 3 alternative free models
        error_msg = ans
        models = self.fetch_all_models()
        if models:
            import random
            # Shuffle models to randomly try others, excluding the one we just tried
            alternatives = [m["id"] for m in models if m["id"] != self.model]
            random.shuffle(alternatives)
            
            original_model = self.model
            for alt_model in alternatives[:3]:
                self.model = alt_model
                ans = self._call(messages)
                if ans and not ans.startswith("[LLM Error"):
                    self.model = original_model # restore original preference
                    return ans
                error_msg = ans
            self.model = original_model # restore original preference
            
        # If all retries failed, return the final error message as text
        return f"Failed to generate response after multiple model attempts. Last error: {error_msg}"
