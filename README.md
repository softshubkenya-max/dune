# DUNE-Ω∞-1900-L

## Distributed Unified Neuro-Emergent Intelligence

```
Learn → Understand → Prove → Simulate → Plan → Explain → Adapt
```

A production-ready cognitive AI system implementing a complete cognitive loop: statistical learning, meaning formation, formal reasoning, simulation, planning, adaptive communication, and continuous feedback. 

DUNE operates as a pure-reasoning **Cognitive Operating System** (DUNE-XΩΩΩ-R). It decouples Large Language Models (LLMs) from the actual reasoning pipeline, relegating the LLM strictly to input parsing and output fluency, while routing internal processing through a structured, deterministic, and explainable cognitive framework.

---

## 🧠 The DUNE-XΩΩΩ-R Architecture

The core of DUNE's cognitive processing flows through the `ReasoningEngine`, which orchestrates several distinct subsystems:

1. **Input LLM (Fluency Layer)**: Takes messy human input and extracts a clean, structured JSON concept representation.
2. **Concept Encoder**: Converts the structured JSON into vector/symbolic formats for ingestion.
3. **Fly Connectome Router**: Simulates the routing function of the Drosophila connectome, determining which cognitive circuits (e.g., security, science, general logic) to activate based on the domain.
4. **ART / SRF (Adaptive Resonance Theory / Semantic Resonance Fields)**: Performs pattern matching, meaning formation, and novelty detection on the encoded concept.
5. **TRIBE Filter**: Evaluates the human relevance of the reasoning graph, filtering facts from the Knowledge Graph to prioritize high human utility.
6. **Memory Complex & World Model**: Integrates the Knowledge Graph, De Bruijn memory (state transitions), Episodic memory, and simulates potential actions and outcomes.
7. **Executive Planner**: Formulates decisions and optimal paths based on the current goal.
8. **Output LLM (Fluency Layer)**: Translates DUNE's raw reasoning trace into a fluent, highly in-depth explanation, often complete with auto-generated Mermaid.js diagrams, D3.js visualizations, or Excalidraw whiteboards.

---

## 🏛️ The Four Pillars

Underpinning the reasoning engine are four foundational pillars:

| Pillar | Component | Function |
|--------|-----------|----------|
| **DUNE-1900** | Statistical Learning Engine | N-grams, TF-IDF, Bayesian Updates, Semantic Graphs |
| **DUNE-ΩΩΩ** | Cognitive Operating System | SRF/ART, De Bruijn Memory, Episodic Memory, World Model, Executive Planning |
| **DUNE-Logic** | Mathematical Reasoning Layer | A* Search, Bayesian Networks, Markov Models, Graph Algorithms, Boolean Logic, Linear Programming, MDPs |
| **DUNE-VI** | Cognitive Publishing Engine | Representation Selection, Knowledge Transformation, Audience Modeling, Adaptive Communication |

---

## 📡 RAG & Autonomous Ingestion

DUNE features a robust **Retrieval-Augmented Generation (RAG) pipeline** backed by an **Autonomous Scheduler**.
- **MCP Data Ingestor**: Autonomously crawls and ingests data from MCP servers, APIs (like arXiv and OpenAlex), URLs, files, and directories.
- **Background Learning**: The scheduler continuously runs in the background, updating DUNE's knowledge structures and Bayesian priors without user intervention.

---

## 🚀 How to Run

DUNE includes a self-contained Web UI and API server (no external dependencies needed besides standard Python libraries).

### 1. Prerequisites

You need an OpenRouter API key for the LLM fluency layers (input parsing and output generation).

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### 2. Start the Server

Run the standalone HTTP server:

```bash
python3 ui/server.py
```

The server will initialize the cognitive engine, start the autonomous background ingestion scheduler, and open an HTTP server on port `8080`.

### 3. Access the Interfaces

Navigate to the following URLs in your browser:
- **Main Client**: `http://localhost:8080/client`
- **Developer/Debug Interface**: `http://localhost:8080/dev`
- **Audio-Visual Interface**: `http://localhost:8080/av`

---

## 💻 Python SDK Quick Start

You can also use DUNE directly in Python without the Web UI:

```python
from dune import DUNE, create_dune

# Create a ready-to-use DUNE instance
dune = create_dune()

# Learn from text
dune.learn("The universe is vast and contains billions of galaxies")

# Reason about a query
result = dune.reason("What is the universe?")
print(result['conclusion'])

# Get an adaptive explanation
print(dune.explain("universe"))

# Plan actions to achieve a goal
plan = dune.plan("research cosmology")
print(f"Plan utility: {plan['total_utility']:.2f}")

# Provide feedback for continuous improvement
dune.provide_feedback("universe", 0.9, "Great explanation!")

# Full reasoning trace
print(dune.explain_with_reasoning("What is DUNE?"))
```

## Running Tests

Verify the integrity of all pillars, mathematical logic layers, and the integrated cognitive system:

```bash
python3 -m dune.tests.test_all
```

## Design Principles

- **No gradient descent**: Uses classical algorithms (Bayesian, TF-IDF, N-grams, A*) for core reasoning.
- **Explainable**: Every decision, semantic resonance, and logical proof can be traced and explained.
- **Continual**: Learning and adaptation over time with feedback and autonomous background ingestion.
- **Low-cost**: Extremely efficient algorithms without GPU requirements for the cognitive core.
- **Provable**: Mathematical verification where possible (Boolean Logic tautologies, Linear Programming, Markov Chains).