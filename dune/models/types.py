"""
DUNE-Ω∞-1900-L Core Data Types
================================
Foundational type definitions for the entire system.
"""

from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    Set, Tuple, TypeVar, Union
)
from enum import Enum, auto
from datetime import datetime
import uuid
import math
import json


# ─────────────────────────── Generic Types ───────────────────────────

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


# ─────────────────────────── Knowledge & Evidence ────────────────────

@dataclass
class Evidence:
    """A piece of evidence retrieved from a knowledge source."""
    content: str
    source: str
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source': self.source,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }


@dataclass
class KnowledgeAtom:
    """Atomic unit of knowledge in the system."""
    fact: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 0.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    relations: List['Relation'] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'fact': self.fact,
            'confidence': self.confidence,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'relations': [r.to_dict() for r in self.relations],
        }


@dataclass
class Relation:
    """A relationship between two knowledge atoms."""
    source_id: str
    target_id: str
    relation_type: str  # e.g., "causes", "implies", "is_a", "part_of"
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'confidence': self.confidence,
            'metadata': self.metadata,
        }


# ─────────────────────────── N-gram Types ────────────────────────────

@dataclass
class NGram:
    """An n-gram with its statistical properties."""
    tokens: Tuple[str, ...]
    count: int = 0
    probability: float = 0.0
    log_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tokens': list(self.tokens),
            'count': self.count,
            'probability': self.probability,
            'log_probability': self.log_probability,
        }


# ─────────────────────────── Semantic Types ──────────────────────────

@dataclass
class SemanticResonanceField:
    """A semantic resonance field (SRF) for meaning formation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    centroid: List[float] = field(default_factory=list)
    radius: float = 1.0
    label: str = ""
    activation: float = 0.0
    resonance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'centroid': self.centroid,
            'radius': self.radius,
            'label': self.label,
            'activation': self.activation,
            'resonance': self.resonance,
        }


@dataclass
class SemanticMap:
    """A map of semantic concepts and their relationships."""
    concepts: Dict[str, SemanticResonanceField] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)
    similarity_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_concept(self, concept: SemanticResonanceField) -> None:
        self.concepts[concept.id] = concept
        if concept.id not in self.adjacency:
            self.adjacency[concept.id] = set()

    def add_edge(self, source_id: str, target_id: str) -> None:
        if source_id in self.adjacency:
            self.adjacency[source_id].add(target_id)
        if target_id in self.adjacency:
            self.adjacency[target_id].add(source_id)


# ─────────────────────────── Memory Types ────────────────────────────

@dataclass
class MemoryState:
    """A state in De Bruijn or episodic memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    strength: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'strength': self.strength,
        }


@dataclass
class Episode:
    """An episodic memory entry."""
    experience: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    outcome: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'experience': self.experience,
            'outcome': str(self.outcome) if self.outcome else None,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
        }


# ─────────────────────────── Planning Types ──────────────────────────

@dataclass
class Action:
    """An action that can be taken by the system."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_reward: float = 0.0
    expected_cost: float = 0.0
    risk: float = 0.0
    confidence: float = 1.0

    def utility(self, risk_aversion: float = 0.5) -> float:
        return (self.expected_reward - self.expected_cost
                - risk_aversion * self.risk)


@dataclass
class Plan:
    """A sequence of actions to achieve a goal."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actions: List[Action] = field(default_factory=list)
    goal: str = ""
    total_utility: float = 0.0
    constraints: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'actions': [a.name for a in self.actions],
            'goal': self.goal,
            'total_utility': self.total_utility,
            'constraints': self.constraints,
        }


# ─────────────────────────── Model States ────────────────────────────

class KnowledgeGraph:
    """Weighted directed graph of knowledge."""
    def __init__(self) -> None:
        self.nodes: Dict[str, KnowledgeAtom] = {}
        self.edges: Dict[str, Dict[str, Relation]] = {}

    def add_node(self, atom: KnowledgeAtom) -> None:
        self.nodes[atom.id] = atom
        if atom.id not in self.edges:
            self.edges[atom.id] = {}

    def add_edge(self, relation: Relation) -> None:
        if relation.source_id not in self.edges:
            self.edges[relation.source_id] = {}
        self.edges[relation.source_id][relation.target_id] = relation

    def get_neighbors(self, node_id: str) -> List[Tuple[str, Relation]]:
        if node_id not in self.edges:
            return []
        return [(t, r) for t, r in self.edges[node_id].items()]

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """BFS shortest path."""
        if source not in self.nodes or target not in self.nodes:
            return None
        visited: Set[str] = {source}
        queue: List[Tuple[str, List[str]]] = [(source, [source])]
        while queue:
            current, path = queue.pop(0)
            if current == target:
                return path
            for neighbor in self.edges.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': {
                s: {t: r.to_dict() for t, r in edges.items()}
                for s, edges in self.edges.items()
            },
        }


@dataclass
class WorldState:
    """Current state of the world model."""
    variables: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    uncertainty: Dict[str, float] = field(default_factory=dict)

    def clone(self) -> 'WorldState':
        return WorldState(
            variables=dict(self.variables),
            timestamp=self.timestamp,
            uncertainty=dict(self.uncertainty),
        )


# ─────────────────────────── VI Types ────────────────────────────────

class DifficultyLevel(Enum):
    BEGINNER = auto()
    INTERMEDIATE = auto()
    EXPERT = auto()


@dataclass
class ConceptProfile:
    """Profile of a concept for representation selection."""
    name: str
    importance: float = 0.5
    difficulty: float = 0.5
    novelty: float = 0.5
    dependencies: List[str] = field(default_factory=list)
    visualizability: float = 0.5
    ambiguity: float = 0.5

    @property
    def entropy(self) -> float:
        return self.ambiguity * max(self.difficulty, 0.1) * max(self.novelty, 0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'importance': self.importance,
            'difficulty': self.difficulty,
            'novelty': self.novelty,
            'dependencies': self.dependencies,
            'visualizability': self.visualizability,
            'ambiguity': self.ambiguity,
            'entropy': self.entropy,
        }


class RepresentationType(Enum):
    TIMELINE = auto()
    FLOWCHART = auto()
    TREE = auto()
    CONCEPT_GRAPH = auto()
    COMPARISON = auto()
    COLLAPSED = auto()
    PRESENTATION = auto()
    STORY = auto()
    ANALOGY = auto()
    DIAGRAM = auto()
    FORMAL = auto()
    DENSE = auto()
    REFERENCE = auto()