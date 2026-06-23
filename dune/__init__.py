"""
DUNE-Ω∞-1900-L: Distributed Unified Neuro-Emergent Intelligence
=================================================================

A production-ready cognitive AI system implementing:
  • DUNE-1900: Statistical Learning Engine
  • DUNE-ΩΩΩ: Cognitive Operating System
  • DUNE-Logic: Mathematical Reasoning Layer
  • DUNE-VI: Cognitive Publishing Engine

Usage:
    from dune import DUNE, create_dune

    dune = create_dune()
    dune.learn("Knowledge is a collection of facts and relationships")
    result = dune.reason("What is knowledge?")
    print(dune.explain("knowledge"))
"""

from dune.orchestrator import DUNE, create_dune
from dune.engine.statistical import (
    DUNE1900, NGramModel, TFIDFVectorizer, BayesianUpdater,
    SemanticGraphBuilder, Tokenizer,
)
from dune.core.cognitive import (
    CognitiveOS, SemanticResonanceEngine, MemoryComplex,
    DeBruijnMemory, EpisodicMemory, WorldModel, ExecutivePlanner,
)
from dune.logic.reasoning import (
    DUNELogic, AStar, BayesianReasoner, MarkovChain,
    HiddenMarkovModel, GraphAlgorithms, BooleanLogic,
    LinearProgram, MDP,
)
from dune.vi.publishing import (
    DUNEVI, RepresentationEngine, KnowledgeTransformer,
    HumanModel, ContentGenerator,
)
from dune.models.types import (
    Evidence, KnowledgeAtom, Relation, KnowledgeGraph,
    SemanticResonanceField, MemoryState, Episode,
    Action, Plan, WorldState,
    ConceptProfile, RepresentationType, DifficultyLevel,
)
from dune.integrations import (
    RAGPipeline, SemanticKernelOrchestrator, BotpressConnector,
    MathRenderer, MarkdownProcessor, VisualizationEngine,
    GenerativeUIComponent, get_capability, list_all_capabilities,
    CapabilityRegistry,
)
from dune.modules import (
    ModulesManager, get_modules_manager, get_module,
    list_modules, get_module_status,
)

__version__ = "1.0.0"
__author__ = "DUNE Research"
__all__ = [
    # Main orchestrator
    "DUNE", "create_dune",
    # DUNE-1900
    "DUNE1900", "NGramModel", "TFIDFVectorizer",
    "BayesianUpdater", "SemanticGraphBuilder", "Tokenizer",
    # DUNE-ΩΩΩ
    "CognitiveOS", "SemanticResonanceEngine", "MemoryComplex",
    "DeBruijnMemory", "EpisodicMemory", "WorldModel", "ExecutivePlanner",
    # DUNE-Logic
    "DUNELogic", "AStar", "BayesianReasoner", "MarkovChain",
    "HiddenMarkovModel", "GraphAlgorithms", "BooleanLogic",
    "LinearProgram", "MDP",
    # DUNE-VI
    "DUNEVI", "RepresentationEngine", "KnowledgeTransformer",
    "HumanModel", "ContentGenerator",
    # Data types
    "Evidence", "KnowledgeAtom", "Relation", "KnowledgeGraph",
    "SemanticResonanceField", "MemoryState", "Episode",
    "Action", "Plan", "WorldState",
    "ConceptProfile", "RepresentationType", "DifficultyLevel",
    # Integrated capabilities
    "RAGPipeline", "SemanticKernelOrchestrator", "BotpressConnector",
    "MathRenderer", "MarkdownProcessor", "VisualizationEngine",
    "GenerativeUIComponent", "get_capability", "list_all_capabilities",
    "CapabilityRegistry",
    # Modules
    "ModulesManager", "get_modules_manager", "get_module",
    "list_modules", "get_module_status",
]