"""Multi-Provider LLM Orchestrator - Automatic fallback between OpenRouter and HuggingFace."""

import os
from typing import List, Dict, Any

class MultiProviderLLM:
    """
    Orchestrates multiple LLM providers with automatic fallback.
    Tries OpenRouter first, falls back to HuggingFace for free models.
    """
    
    def __init__(self):
        from dune.llm.openrouter import OpenRouterClient
        from dune.llm.huggingface import HuggingFaceClient
        
        self.openrouter = OpenRouterClient()
        self.huggingface = HuggingFaceClient()
        self.current_provider = 'openrouter'  # Start with OpenRouter
        self.last_error = None
    
    def format_response(self, user_input: str, dune_trace: str) -> str:
        """
        Format DUNE's reasoning using available LLM providers.
        Automatically switches between providers on failure.
        
        Args:
            user_input: Original user query
            dune_trace: Raw DUNE reasoning output
            
        Returns:
            Formatted response with automatic fallback
        """
        # Try primary provider first
        response = self._try_provider(self.current_provider, user_input, dune_trace)
        if not response.startswith("[LLM Error") and not response.startswith("[HuggingFace"):
            return response
        
        # Switch to fallback if primary fails
        fallback_provider = 'huggingface' if self.current_provider == 'openrouter' else 'openrouter'
        response = self._try_provider(fallback_provider, user_input, dune_trace)
        
        if not response.startswith("[LLM Error") and not response.startswith("[HuggingFace"):
            # Success with fallback - update primary
            self.current_provider = fallback_provider
            return response
        
        # Both failed - return combined error
        return f"{response}\n[Fallback: {self.current_provider} also unavailable]"
    
    def _try_provider(self, provider: str, user_input: str, dune_trace: str) -> str:
        """Try a specific provider."""
        try:
            if provider == 'openrouter':
                return self.openrouter.format_response(user_input, dune_trace)
            elif provider == 'huggingface':
                return self.huggingface.format_response(user_input, dune_trace)
        except Exception as e:
            self.last_error = str(e)
        
        return f"[{provider.upper()} Error: Unable to generate response]"
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """Get models from all providers."""
        models = {
            'openrouter': [],
            'huggingface': self.huggingface.list_free_models(),
        }
        
        try:
            or_models = self.openrouter.fetch_all_models()
            models['openrouter'] = [m.get('id', '') for m in or_models if m.get('id')]
        except:
            pass
        
        return models
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'orchestrator': 'Multi-Provider LLM',
            'primary_provider': self.current_provider,
            'openrouter_status': self.openrouter.get_status() if hasattr(self.openrouter, 'get_status') else 'Unknown',
            'huggingface_status': self.huggingface.get_status(),
            'last_error': self.last_error,
            'auto_fallback': 'enabled',
        }
    
    def switch_provider(self, provider: str) -> bool:
        """Manually switch provider."""
        if provider in ['openrouter', 'huggingface']:
            self.current_provider = provider
            return True
        return False


# Singleton instance
_multi_provider = None

def get_multi_provider_llm() -> MultiProviderLLM:
    """Get or create multi-provider LLM instance."""
    global _multi_provider
    if _multi_provider is None:
        _multi_provider = MultiProviderLLM()
    return _multi_provider
