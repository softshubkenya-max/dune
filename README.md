# DUNE-Ω∞-XΩΩ-1900-L — Cognitive Operating System for Autonomous Discovery & Invention

## Complete End-to-End System Reference: Input → Processing → Output

---

# TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [Complete Architecture Diagram](#2-complete-architecture-diagram)
3. [Software Stack: Input to Output](#3-software-stack-input-to-output)
4. [Step-by-Step Pipeline: Query to Response](#4-step-by-step-pipeline-query-to-response)
   - [Step 1: HTTP Request Handling](#step-1-http-request-handling-uiserverpy)
   - [Step 2: LLM Query Extraction](#step-2-llm-query-extraction)
   - [Step 3: Reasoning Engine Pipeline](#step-3-reasoning-engine-pipeline)
   - [Step 4: Explanation Generation](#step-4-explanation-generation-duneorchestratorpy)
   - [Step 5: Skills Context Injection](#step-5-skills-context-injection)
   - [Step 6: Trace Formatting](#step-6-trace-formatting)
   - [Step 7: LLM Response Generation](#step-7-llm-response-generation)
   - [Step 8: Response Delivery](#step-8-response-delivery)
5. [Module Reference](#5-module-reference)
6. [Data Structures](#6-data-structures)
7. [LLM Provider Integration](#7-llm-provider-integration)
8. [API Reference](#8-api-reference)
9. [Configuration](#9-configuration)
10. [Testing & Verification](#10-testing--verification)

---

# 1. SYSTEM OVERVIEW

DUNE (Distributed Unified Neuro-Emergent Intelligence) is a **production-ready cognitive AI platform** that transforms natural language queries into comprehensive, structured discovery reports through a multi-stage pipeline — from concept encoding through formal reasoning, memory retrieval, multi-pillar synthesis, and LLM-powered response generation.

## 1.1 Core Capabilities

- **Learn**: Ingest knowledge from text, build statistical models, semantic graphs, and Bayesian priors
- **Reason**: Multi-pillar inference combining statistical, cognitive, and formal reasoning
- **Plan**: Generate and verify executable action plans with simulation
- **Explain**: Produce human-ready, audience-adapted explanations
- **Discover**: Autonomous hypothesis generation, invention search, and scientific breakthrough discovery

## 1.2 Four Pillars

| Pillar | Location | Purpose |
|--------|----------|---------|
| **DUNE-1900** | `dune/engine/statistical.py` | Statistical Learning Engine (tokenization, n-grams, TF-IDF, Bayesian updating) |
| **DUNE-ΩΩΩ** | `dune/core/cognitive.py` | Cognitive OS (semantic resonance, memory systems, world model, executive planning) |
| **DUNE-Logic** | `dune/logic/reasoning.py` | Formal Reasoning (A* search, Bayesian inference, Markov chains, MDP, Boolean logic, graph algorithms) |
| **DUNE-VI** | `dune/vi/publishing.py` | Cognitive Publishing (representation selection, explanation generation, human modeling) |

## 1.3 Orchestration Layer

- **Reasoning Engine** (`dune/core/reasoning_engine.py`): Master orchestrator for the DUNE-XΩΩΩ-R flow
  - Pipeline: Concept Encoder → Fly Connectome → ART/SRF → TRIBE → KG/De Bruijn/World Model/Planner → Decision
- **Multi-Provider LLM** (`dune/llm/multi_provider.py`): Automatic fallback between Ollama, OpenRouter, HuggingFace
- **Discovery Prompt** (`dune/llm/discovery_prompt.py`): Loads and compiles the Universal Discovery & Invention Engine template
- **Skills Module** (`dune/modules_anthropic_skills.py`): Injects 17 Anthropic Skills for output structuring

---

# 2. COMPLETE ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Browser / API Client)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ POST /api/chat  { query: "..." }
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ui/server.py — DUNEAPIHandler                                       │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  1. extract_query(user_input) → JSON concept                    │   │
│  │     (LLM converts messy human language to structured JSON)     │   │
│  │                                                                  │   │
│  │  2. reasoning_engine.process(json) → decision_trace             │   │
│  │     ┌───────────────────────────────────────────────────────┐   │   │
│  │     │  a. ConceptEncoder.encode(json) → vector + concept    │   │   │
│  │     │  b. FlyConnectomeRouter.route(vector) → circuits      │   │   │
│  │     │  c. SemanticEngine.activate(vector) → field + score   │   │   │
│  │     │  d. TribeFilter.filter(facts, goal) → filtered_facts  │   │   │
│  │     │  e. Memory.query_facts(concept) → raw_facts           │   │   │
│  │     │  f. Planner.add_goal(goal) → plan context             │   │   │
│  │     │  g. → decision_trace                                  │   │   │
│  │     └───────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  │  3. dune.explain_with_reasoning(concept) → VI explanation       │   │
│  │  4. inject get_skills_context() → decision_trace                │   │
│  │  5. format_decision_for_llm(trace) → formatted_trace (JSON)    │   │
│  │  6. llm_client.format_response(query, formatted_trace)          │   │
│  │     → discovery_format_response()                               │   │
│  │       ├── loads system prompt template                           │   │
│  │       ├── injects skills context block                           │   │
│  │       └── returns [system_msg, user_msg]                         │   │
│  │  7. LLM generates response (Ollama→OpenRouter→HF auto-fallback) │   │
│  │  8. Response sent to client (plain text or SSE stream)          │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

# 3. END-TO-END PIPELINE (Input → Output)

## Step 1: User Input

The user sends a POST request to `/api/chat` with a JSON body:
```json
{
  "query": "How would you design a more efficient solar cell?",
  "session_id": "abc123"
}
```

## Step 2: Query Extraction (`ui/server.py` → `dune/llm/openrouter.py`)

The `_handle_chat` method in `DUNEAPIHandler` calls `llm_client.extract_query(query)`.

This sends the user's messy human language to an LLM with a system prompt that instructs it to output ONLY valid JSON:
```json
{
  "domain": "energy",
  "object": "solar cell",
  "goal": "design more efficient solar cell",
  "environment": "research"
}
```

If the LLM is unavailable, it falls back to `{"object": "<user_input>"}`.

## Step 3: Reasoning Pipeline (`dune/core/reasoning_engine.py`)

`reasoning_engine.process(core_query_json)` runs the DUNE-XΩΩΩ-R pipeline:

### 3a. Concept Encoding (`dune/core/concept_encoder.py`)
- Takes the JSON input
- Encodes the concept into a normalized vector representation
- Extracts the core concept, domain, and structured metadata

### 3b. Fly Connectome Routing (`dune/core/fly_connectome.py`)
- Routes the encoded vector through sparse neural circuits
- Returns activated circuit paths that guide reasoning direction

### 3c. Semantic Resonance Activation (`dune/core/cognitive.py` → `SemanticResonanceEngine`)
- Activates semantic resonance fields using Adaptive Resonance Theory
- `activate(vector)` → returns `(field_id, resonance_score)`
- If no existing field matches, `learn(vector, concept)` creates a new semantic field

### 3d. Memory Query (`dune/core/cognitive.py` → `DeBruijnMemory` / `EpisodicMemory`)
- Queries memory for facts related to the concept
- Returns raw facts from knowledge graph

### 3e. TRIBE Filter (`dune/core/tribe.py`)
- Filters memory facts based on the goal context
- Returns only the most relevant facts for the current reasoning task

### 3f. Goal Planning (`dune/core/cognitive.py` → `ExecutivePlanner`)
- The goal from the structured query is added to the planner
- This prepares the planning context for downstream use

### 3g. Decision Trace Output

The pipeline produces a `decision_trace` dict:
```json
{
  "concept": "solar cell",
  "domain": "energy",
  "activated_circuits": [...],
  "art_resonance_score": 0.8734,
  "tribe_filtered_facts": [
    "Photovoltaic cells convert light to electricity via the photoelectric effect",
    "Silicon-based cells have ~22% efficiency ceiling due to bandgap limitations",
    "Perovskite materials show promise with 25%+ efficiency in lab settings"
  ],
  "goal": "design more efficient solar cell"
}
```

## Step 4: DUNE-VI Explanation Generation (`dune/orchestrator.py` → `explain_with_reasoning`)

`dune.explain_with_reasoning(concept)`:
- Uses the concept to query DUNE-VI publishing engine
- The `RepresentationEngine` chooses the optimal explanation format
- The `KnowledgeTransformer` renders the explanation
- Returns a rich explanation string that gets added to the trace as `dune_vi_explanation`

## Step 5: Skills Context Injection

Before formatting the trace for the LLM, `ui/server.py` injects `decision_trace["skills_context"] = get_skills_context()`.

This adds the full Anthropic Skills listing (17 skills across 5 categories) into the trace, so the LLM knows which skills to activate for structuring its output.

## Step 6: Trace Formatting

`reasoning_engine.format_decision_for_llm(trace)` serializes the enriched trace as a JSON string:
```json
{
  "concept": "solar cell",
  "domain": "energy",
  "activated_circuits": [...],
  "art_resonance_score": 0.8734,
  "tribe_filtered_facts": [...],
  "goal": "design more efficient solar cell",
  "dune_vi_explanation": "...",
  "skills_context": "## Available Skills\n\n🎨 Design & Creative: ...\n..."
}
```

## Step 7: LLM Response Generation

`llm_client.format_response(user_input, formatted_trace)` is called. This:

### Step 7a: Load Discovery Prompt Template
`discovery_format_response()` in `dune/llm/discovery_prompt.py`:
1. Loads the template from `prompts/universal_discovery_invention_engine.md`
2. Replaces `[INSERT PROBLEM]` with the user's query
3. Builds the system message (concise, ~739 tokens)
4. Builds the user message with:
   - Target problem
   - DUNE context (the formatted trace)
   - Skills context block (compact listing of all 17 skills)
   - Final instruction: "Produce a complete discovery report... Be conversational, in-depth, and concrete."

### Step 7b: Provider Fallback Chain
The `MultiProviderLLM` tries providers in order:
1. **Ollama** (local, no API key needed) — fastest, offline
2. **OpenRouter** (online, needs API key) — best free models
3. **HuggingFace** (online, needs API key) — fallback

If one fails, it automatically tries the next.

## Step 8: Response Delivery

The formatted response is:
- Stored in the chat session database (`dune/db/chat_store.py`)
- Returned to the client as plain text (non-streaming) or SSE chunks (streaming)
- For SSE streaming, each word is sent as a separate event with index tracking

---

# 9. CONFIGURATION

## 9.1 Environment Variables

Create a \`.env\` file in the project root:

```bash
# ─── LLM Provider Configuration ──────────────────────────────

# OpenRouter (cloud, online)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free

# HuggingFace (cloud, online)
HF_API_TOKEN=your_huggingface_token_here

# Ollama (local, fastest)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=neural-chat  # or mistral, llama2, etc.

# ─── Server Configuration ────────────────────────────────────

# Server binding
DUNE_HOST=0.0.0.0
DUNE_PORT=8080

# ─── Feature Flags ───────────────────────────────────────────

# Autonomous ingestion
ARXIV_ENABLED=true
OPENALEX_ENABLED=true
PUBMED_ENABLED=true
```

## 9.2 MCP Ingestion Sources

The \`AutonomousScheduler\` in \`ui/server.py\` pre-configures these sources:

**Academic APIs:**
- arXiv: quantum, AI, biology, math, physics
- OpenAlex: AI, physics
- Crossref: general science
- PLOS: general science
- PubMed: health

**Source config format:**
```python
scheduler.ingestor.add_api_source(
    name="Source_Name",
    url="https://api.example.com/endpoint",
    # optional: headers, parser_type, interval_minutes
)
```

## 9.3 LLM Provider Selection

**Priority order (automatic fallback):**
1. **Ollama** (local) — if \`OLLAMA_URL\` is reachable
2. **OpenRouter** (cloud) — if \`OPENROUTER_API_KEY\` is set
3. **HuggingFace** (cloud) — if \`HF_API_TOKEN\` is set

**Manual selection:**
```python
from dune.llm.multi_provider import get_multi_provider_llm
llm = get_multi_provider_llm()
llm.switch_provider('openrouter')
llm.set_model('openrouter', 'meta-llama/llama-3-3-70b-instruct:free')
```

## 9.4 Server Startup

```bash
# Start the HTTP server
bash start_server.sh

# Or manually:
cd /home/ty/Documents/bud
source venv/bin/activate
python ui/server.py

# Server runs at http://localhost:8080 (or next available port)
```
# dune
# dune
