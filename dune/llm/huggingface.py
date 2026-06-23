"""HuggingFace Inference API Client - Free and open-source models."""

import requests
import os
from typing import List, Dict, Any, Optional

class HuggingFaceClient:
    """Client for HuggingFace Inference API with free model access."""
    
    # Free models available on HuggingFace with good performance
    FREE_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
        "tiiuae/falcon-7b-instruct",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    ]
    
    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or os.environ.get('HUGGINGFACE_API_KEY', '')
        self.model = model or self.FREE_MODELS[0]
        self.base_url = "https://api-inference.huggingface.co/models"
        self.available = bool(self.api_key)
        self.current_model_idx = 0
    
    def format_response(self, user_input: str, dune_trace: str) -> str:
        """
        Format DUNE's raw reasoning into fluent explanation using HuggingFace.
        
        Args:
            user_input: Original user query
            dune_trace: Raw DUNE reasoning output
            
        Returns:
            Formatted response or error message
        """
        if not self.available:
            return "[HuggingFace: API key not configured]"
        
        messages = [
            {
                "role": "system",
                "content": "You are an eccentric but brilliant AI scientist. Explain DUNE's reasoning with energy and personality. Use interjections like 'Ah!', 'Hmm...', 'Interesting!'. Be quick-witted and playful."
            },
            {
                "role": "user",
                "content": f"User asked: {user_input}\n\nDUNE's reasoning:\n{dune_trace}\n\nExplain this in an engaging, scientific way."
            }
        ]
        
        return self._call(messages)
    
    def _call(self, messages: List[Dict]) -> str:
        """Call HuggingFace API with automatic fallback."""
        if not self.available:
            return "[HuggingFace not configured]"
        
        # Try current model and alternates
        for attempt in range(len(self.FREE_MODELS)):
            model = self.FREE_MODELS[self.current_model_idx % len(self.FREE_MODELS)]
            
            try:
                response = requests.post(
                    f"{self.base_url}/{model}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "inputs": self._format_chat(messages),
                        "parameters": {
                            "max_new_tokens": 500,
                            "temperature": 0.8,
                            "top_p": 0.95,
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('generated_text', '')
                        if text:
                            self.model = model
                            return text.strip()
                
                # Try next model on failure
                self.current_model_idx += 1
                continue
                
            except Exception as e:
                self.current_model_idx += 1
                continue
        
        return "[HuggingFace: All models currently unavailable]"
    
    def _format_chat(self, messages: List[Dict]) -> str:
        """Format messages for HuggingFace chat model."""
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            formatted += f"{role}: {content}\n\n"
        formatted += "ASSISTANT:"
        return formatted
    
    def list_free_models(self) -> List[str]:
        """Get list of available free models."""
        return self.FREE_MODELS
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return {
            'provider': 'HuggingFace',
            'available': self.available,
            'current_model': self.model,
            'models': self.FREE_MODELS,
            'api_configured': bool(self.api_key),
        }
