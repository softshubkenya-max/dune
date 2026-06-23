"""
Multi-Provider LLM Integration for DUNE
========================================

DUNE now supports automatic switching between multiple LLM providers for 
uninterrupted operation. When one provider experiences rate limits or errors, 
the system automatically switches to available alternatives.

Supported Providers
===================

1. OpenRouter
   - Access to 40+ models
   - Free tier available
   - Models: Claude, GPT, Mistral, Llama, etc.
   - Setup: export OPENROUTER_API_KEY=your_key

2. HuggingFace Inference API
   - Free access to open-source models
   - No API key required (but recommended for higher limits)
   - Models:
     • Mistral-7B-Instruct
     • Llama-2-7B-Chat
     • Falcon-7B-Instruct
     • Nous-Hermes-2-Mixtral-8x7B
   - Setup: export HUGGINGFACE_API_KEY=your_token (optional)

Automatic Fallback Behavior
============================

1. Primary Provider (OpenRouter) is tried first
2. If it returns an error (429 rate limit, 402 payment, etc.):
   - System automatically switches to HuggingFace
3. If both fail:
   - Returns combined error message
   - Logs the issue for diagnostics

Configuration
==============

Environment Variables:
  - OPENROUTER_API_KEY: OpenRouter API key (optional)
  - HUGGINGFACE_API_KEY: HuggingFace token (optional)

API Endpoints:
  - GET /api/config/openrouter - Get current provider status
  - POST /api/config/openrouter - Switch provider
    Example: {"provider": "huggingface"}
  - GET /api/models - List all available models from all providers

Usage
=====

Python:
    from dune.llm.multi_provider import get_multi_provider_llm
    
    llm = get_multi_provider_llm()
    
    # Automatically tries OpenRouter first, then HuggingFace
    response = llm.format_response(user_query, dune_reasoning)
    
    # Check status
    status = llm.get_status()
    print(status['primary_provider'])  # Current provider
    
    # Manually switch
    llm.switch_provider('huggingface')

REST API:
    # Get provider status
    curl http://localhost:8081/api/config/openrouter
    
    # Switch to HuggingFace
    curl -X POST http://localhost:8081/api/config/openrouter \
      -H 'Content-Type: application/json' \
      -d '{"provider": "huggingface"}'
    
    # List all models
    curl http://localhost:8081/api/models

Troubleshooting
===============

Q: Getting "API key not configured" errors?
A: The system works without API keys but with rate limits. 
   Set HUGGINGFACE_API_KEY for more requests.

Q: Models keep switching?
A: This means both providers are experiencing issues. Check:
   - Internet connectivity
   - API rate limits
   - API key validity
   - Provider service status

Q: How to force use one provider?
A: Use the API endpoint:
    curl -X POST http://localhost:8081/api/config/openrouter \
      -d '{"provider": "openrouter"}'

Performance Notes
=================

- OpenRouter models: Faster, higher quality, requires API key
- HuggingFace models: Free, slightly slower, good for most tasks
- Total latency with fallback: ~2-5 seconds worse case
- Automatic switching happens transparently to user

Model Characteristics
====================

OpenRouter (when available):
  - Quality: ★★★★★
  - Speed: ★★★★★
  - Cost: Free tier limited
  - Switching latency: <500ms

HuggingFace (free tier):
  - Quality: ★★★★☆
  - Speed: ★★★★☆
  - Cost: Free
  - Switching latency: 1-2s

Future Enhancements
===================

- [ ] Add Replicate provider support
- [ ] Add Together AI provider
- [ ] Local model inference (ollama)
- [ ] Provider health monitoring dashboard
- [ ] Cost tracking across providers
- [ ] Smart provider selection based on query type
