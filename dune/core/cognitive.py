"""
DUNE-ΩΩΩ: Cognitive Operating System
======================================
Converts knowledge into structured thought.
Core: Semantic Core (SRF/ART), Memory Complex (KG, De Bruijn, Episodic),
World Model (Simulation), Executive Planner.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
import math
import random
import uuid
from datetime import datetime
import heapq

from dune.models.types import (
    KnowledgeAtom, KnowledgeGraph, Relation,
    SemanticResonanceField, SemanticMap,
    MemoryState, Episode,
    Action, Plan, WorldState,
    Evidence
)


# ═══════════════════════════════════════════════════════════════════════
# I. SEMANTIC CORE
# ═══════════════════════════════════════════════════════════════════════

class SemanticResonanceEngine:
    """
    Semantic Resonance Fields (SRF) with Adaptive Resonance Theory (ART).
    Forms meaning through pattern matching and resonance detection.
    """

    def __init__(self, learning_rate: float = 0.1, vigilance: float = 0.8):
        self.learning_rate = learning_rate
        self.vigilance = vigilance
        self.fields: Dict[str, SemanticResonanceField] = {}
        self.semantic_map = SemanticMap()

    def create_field(self, label: str, centroid: Optional[List[float]] = None) -> SemanticResonanceField:
        """Create a new semantic resonance field."""
        field = SemanticResonanceField(
            centroid=centroid or [],
            label=label,
        )
        self.fields[field.id] = field
        self.semantic_map.add_concept(field)
        return field

    def activate(self, input_vector: List[float]) -> Tuple[Optional[str], float]:
        """
        Activate fields with input. Returns (field_id, resonance) of best match.
        Uses ART-like vigilance matching.
        """
        best_field_id = None
        best_resonance = 0.0

        for fid, field in self.fields.items():
            if not field.centroid:
                continue

            resonance = self._compute_resonance(input_vector, field.centroid)
            if resonance > self.vigilance and resonance > best_resonance:
                best_resonance = resonance
                best_field_id = fid

        return best_field_id, best_resonance

    def learn(self, input_vector: List[float], label: Optional[str] = None) -> str:
        """
        Learn a new input pattern. Either matches to existing field or creates new one.
        Returns the field ID that learned the pattern.
        """
        best_id, resonance = self.activate(input_vector)

        if best_id and resonance >= self.vigilance:
            # Update matched field
            field = self.fields[best_id]
            for i in range(min(len(input_vector), len(field.centroid))):
                field.centroid[i] += self.learning_rate * (input_vector[i] - field.centroid[i])
            field.activation = resonance
            field.resonance = resonance
            return best_id
        else:
            # Create new field
            field = self.create_field(label or f"concept_{len(self.fields)}")
            field.centroid = input_vector.copy()
            field.activation = 1.0
            return field.id

    def merge_fields(self, field_id_a: str, field_id_b: str) -> Optional[str]:
        """Merge two semantic fields into one."""
        if field_id_a not in self.fields or field_id_b not in self.fields:
            return None

        fa = self.fields[field_id_a]
        fb = self.fields[field_id_b]

        merged_label = f"{fa.label}_{fb.label}" if fa.label and fb.label else fa.label or fb.label
        merged_centroid = [
            (a + b) / 2 for a, b in zip(fa.centroid, fb.centroid)
        ] if fa.centroid and fb.centroid else (fa.centroid or fb.centroid)

        merged = self.create_field(merged_label, merged_centroid)

        # Link in semantic map
        self.semantic_map.add_edge(merged.id, field_id_a)
        self.semantic_map.add_edge(merged.id, field_id_b)

        return merged.id

    def detect_novelty(self, input_vector: List[float]) -> float:
        """Detect how novel an input is relative to existing fields."""
        if not self.fields:
            return 1.0

        max_resonance = 0.0
        for field in self.fields.values():
            if field.centroid:
                resonance = self._compute_resonance(input_vector, field.centroid)
                max_resonance = max(max_resonance, resonance)

        return 1.0 - max_resonance

    def get_semantic_similarity(self, field_id_a: str, field_id_b: str) -> float:
        """Compute semantic similarity between two fields."""
        if field_id_a not in self.fields or field_id_b not in self.fields:
            return 0.0
        fa = self.fields[field_id_a]
        fb = self.fields[field_id_b]
        if not fa.centroid or not fb.centroid:
            return 0.0
        return self._compute_resonance(fa.centroid, fb.centroid)

    def _compute_resonance(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity resonance between two vectors."""
        if not vec_a or not vec_b:
            return 0.0
        min_len = min(len(vec_a), len(vec_b))
        dot = sum(vec_a[i] * vec_b[i] for i in range(min_len))
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a[:min_len]))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b[:min_len]))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def to_dict(self) -> Dict:
        return {
            'learning_rate': self.learning_rate,
            'vigilance': self.vigilance,
            'fields': {k: v.to_dict() for k, v in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticResonanceEngine':
        engine = cls(learning_rate=data['learning_rate'], vigilance=data['vigilance'])
        for fid, fd in data['fields'].items():
            field = SemanticResonanceField(
                id=fid,
                centroid=fd['centroid'],
                radius=fd['radius'],
                label=fd['label'],
                activation=fd['activation'],
                resonance=fd['resonance'],
            )
            engine.fields[fid] = field
            engine.semantic_map.add_concept(field)
        return engine


# ═══════════════════════════════════════════════════════════════════════
# II. MEMORY COMPLEX
# ═══════════════════════════════════════════════════════════════════════

class DeBruijnMemory:
    """
    De Bruijn sequence memory for state transitions.
    State → State → State transitions with reconciliation.
    """

    def __init__(self, order: int = 2):
        self.order = order
        self._states: Dict[str, MemoryState] = {}
        self._transitions: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        self._transition_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

    def add_state(self, state: MemoryState) -> str:
        """Add a state to memory."""
        self._states[state.id] = state
        return state.id

    def record_transition(self, *state_ids: str) -> None:
        """Record a sequence of state transitions."""
        if len(state_ids) != self.order + 1:
            raise ValueError(f"Expected {self.order + 1} states, got {len(state_ids)}")

        key = tuple(state_ids[:-1])
        self._transitions[key].append(state_ids[-1])
        self._transition_counts[key] += 1

        # Strengthen the states
        for sid in state_ids:
            if sid in self._states:
                self._states[sid].strength = min(2.0, self._states[sid].strength + 0.1)

    def predict_next(self, *context_states: str) -> Optional[str]:
        """Predict the next state given context."""
        if len(context_states) < self.order:
            # Pad with None-like handling
            return None

        key = tuple(context_states[-self.order:])
        if key in self._transitions:
            transitions = self._transitions[key]
            # Weight by transition count
            from collections import Counter
            counter = Counter(transitions)
            most_common = counter.most_common(1)
            if most_common:
                return most_common[0][0]

        return None

    def get_transition_probability(self, from_states: Tuple[str, ...], to_state: str) -> float:
        """Get probability of transitioning to to_state from from_states."""
        key = from_states
        total = self._transition_counts.get(key, 0)
        if total == 0:
            return 0.0

        count = sum(1 for t in self._transitions.get(key, []) if t == to_state)
        return count / total

    def path_between(self, start_id: str, end_id: str, max_depth: int = 10) -> Optional[List[str]]:
        """Find a path between two states using De Bruijn transitions."""
        from collections import deque
        visited: Set[str] = {start_id}
        queue = deque([(start_id, [start_id])])

        while queue:
            current, path = queue.popleft()
            if current == end_id:
                return path

            if len(path) >= max_depth:
                continue

            # Look for transitions from current state
            for key, targets in self._transitions.items():
                if current in key:
                    for target in targets:
                        if target not in visited:
                            visited.add(target)
                            queue.append((target, path + [target]))

        return None

    def decay(self, factor: float = 0.99) -> None:
        """Apply decay to all states and transitions."""
        for state in self._states.values():
            state.strength *= factor

        for key in list(self._transition_counts.keys()):
            self._transition_counts[key] = max(0, int(self._transition_counts[key] * factor))

    def to_dict(self) -> Dict:
        # Convert tuple keys to string representation for JSON compatibility
        serialized_transitions = {}
        for k, v in self._transition_counts.items():
            serialized_transitions[',,,join,,,'.join(k)] = v
        return {
            'order': self.order,
            'states': {k: v.to_dict() for k, v in self._states.items()},
            'transition_counts': serialized_transitions,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DeBruijnMemory':
        mem = cls(order=data['order'])
        for sid, sd in data['states'].items():
            state = MemoryState(id=sid, content=sd['content'], strength=sd['strength'])
            state.timestamp = datetime.fromisoformat(sd['timestamp'])
            mem._states[sid] = state
        for key_str, count in data['transition_counts'].items():
            key = tuple(key_str.split(',,,join,,,'))
            mem._transition_counts[key] = count
        return mem


class EpisodicMemory:
    """
    Episodic memory: stores experiences and their outcomes.
    Experience → Outcome with importance weighting.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._episodes: Dict[str, Episode] = {}
        self._index: Dict[str, List[str]] = defaultdict(list)  # keyword → episode IDs

    def store(self, experience: str, outcome: Any = None,
              context: Optional[Dict] = None, importance: float = 0.5) -> str:
        """Store a new episodic memory."""
        if len(self._episodes) >= self.capacity:
            # Evict least important
            oldest_id = min(self._episodes, key=lambda k: (
                self._episodes[k].importance,
                self._episodes[k].timestamp
            ))
            del self._episodes[oldest_id]

        episode = Episode(
            experience=experience,
            outcome=outcome,
            context=context or {},
            importance=importance,
        )
        self._episodes[episode.id] = episode

        # Index keywords
        for word in experience.lower().split():
            if len(word) > 3:
                self._index[word].append(episode.id)

        return episode.id

    def recall(self, query: str, top_k: int = 5) -> List[Episode]:
        """Recall episodes matching the query."""
        query_words = set(w for w in query.lower().split() if len(w) > 3)
        if not query_words:
            return []

        scores: Dict[str, float] = defaultdict(float)
        for word in query_words:
            for ep_id in self._index.get(word, []):
                scores[ep_id] += 1.0

        # Boost by importance
        for ep_id in scores:
            if ep_id in self._episodes:
                scores[ep_id] *= self._episodes[ep_id].importance

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self._episodes[ep_id] for ep_id, _ in ranked if ep_id in self._episodes]

    def recall_recent(self, n: int = 10) -> List[Episode]:
        """Recall most recent episodes."""
        sorted_eps = sorted(
            self._episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        return sorted_eps[:n]

    def recall_important(self, threshold: float = 0.7) -> List[Episode]:
        """Recall episodes above an importance threshold."""
        return [
            ep for ep in self._episodes.values()
            if ep.importance >= threshold
        ]

    def retrieve_evidence(self, query: str, top_k: int = 5) -> List[Evidence]:
        """Retrieve episodes as evidence for a query."""
        episodes = self.recall(query, top_k=top_k)
        return [
            Evidence(
                content=ep.experience,
                source=f"episodic/{ep.id}",
                relevance_score=ep.importance,
                metadata={'outcome': str(ep.outcome), 'context': ep.context},
            )
            for ep in episodes
        ]

    def to_dict(self) -> Dict:
        return {
            'capacity': self.capacity,
            'episodes': {k: v.to_dict() for k, v in self._episodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EpisodicMemory':
        mem = cls(capacity=data['capacity'])
        for ep_id, ep_data in data['episodes'].items():
            ep = Episode(
                experience=ep_data['experience'],
                outcome=ep_data.get('outcome'),
                context=ep_data.get('context', {}),
                importance=ep_data.get('importance', 0.5),
            )
            ep.id = ep_id
            ep.timestamp = datetime.fromisoformat(ep_data['timestamp'])
            mem._episodes[ep_id] = ep
            for word in ep.experience.lower().split():
                if len(word) > 3:
                    mem._index[word].append(ep_id)
        return mem


class MemoryComplex:
    """
    Integrated memory system combining Knowledge Graph, De Bruijn, and Episodic memory.
    """

    def __init__(self):
        self.semantic_memory = KnowledgeGraph()
        self.debruijn_memory = DeBruijnMemory(order=2)
        self.episodic_memory = EpisodicMemory()
        self._attention_weights: Dict[str, float] = defaultdict(float)

    def store_fact(self, fact: str, confidence: float = 1.0,
                   source: str = "memory", relations: Optional[List[Relation]] = None) -> str:
        """Store a fact in semantic memory."""
        atom = KnowledgeAtom(
            fact=fact,
            confidence=confidence,
            source=source,
        )
        if relations:
            atom.relations = relations
        self.semantic_memory.add_node(atom)
        return atom.id

    def store_experience(self, experience: str, outcome: Any = None,
                         importance: float = 0.5) -> str:
        """Store an experience in episodic memory."""
        return self.episodic_memory.store(experience, outcome, importance=importance)

    def store_transition(self, *states: str) -> None:
        """Record a state transition in De Bruijn memory."""
        self.debruijn_memory.record_transition(*states)

    def query_facts(self, query: str, top_k: int = 10) -> List[KnowledgeAtom]:
        """Query semantic memory for facts matching query."""
        query_words = set(query.lower().split())
        scored: List[Tuple[float, KnowledgeAtom]] = []

        for node in self.semantic_memory.nodes.values():
            fact_words = set(node.fact.lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(fact_words))
                score *= node.confidence
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def consolidate(self) -> None:
        """Consolidate memories: strengthen frequently accessed patterns."""
        # Apply decay to De Bruijn memory
        self.debruijn_memory.decay(factor=0.995)

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            'semantic_nodes': len(self.semantic_memory.nodes),
            'semantic_edges': sum(len(e) for e in self.semantic_memory.edges.values()),
            'debruijn_states': len(self.debruijn_memory._states),
            'debruijn_transitions': len(self.debruijn_memory._transitions),
            'episodes': len(self.episodic_memory._episodes),
        }


# ═══════════════════════════════════════════════════════════════════════
# III. WORLD MODEL
# ═══════════════════════════════════════════════════════════════════════

class SimulationStep:
    """A single step in a world simulation."""
    def __init__(self, state: WorldState, action: str, outcome: str,
                 reward: float = 0.0, probability: float = 1.0):
        self.state = state
        self.action = action
        self.outcome = outcome
        self.reward = reward
        self.probability = probability


class WorldModel:
    """
    World model for simulating actions and predicting outcomes.
    Action → Simulation → Future Outcome with counterfactual support.
    """

    def __init__(self, uncertainty_model: Optional[Callable] = None):
        self._causal_rules: List[Dict[str, Any]] = []
        self._state_transition_model: Dict[str, Dict[str, Tuple[str, float]]] = {}
        self._current_state = WorldState()
        self.uncertainty_model = uncertainty_model
        self._simulation_history: List[SimulationStep] = []

    def set_state(self, state: WorldState) -> None:
        """Set the current world state."""
        self._current_state = state.clone()

    def get_state(self) -> WorldState:
        """Get current world state."""
        return self._current_state.clone()

    def add_causal_rule(self, antecedent: str, consequent: str,
                        probability: float = 1.0, delay: int = 0) -> None:
        """Add a causal rule: if antecedent then consequent."""
        self._causal_rules.append({
            'antecedent': antecedent,
            'consequent': consequent,
            'probability': probability,
            'delay': delay,
        })

    def add_transition(self, state_key: str, action: str,
                       next_state_key: str, probability: float = 1.0) -> None:
        """Add a state transition model."""
        if state_key not in self._state_transition_model:
            self._state_transition_model[state_key] = {}
        self._state_transition_model[state_key][action] = (next_state_key, probability)

    def simulate(self, action: Action, steps: int = 1) -> List[SimulationStep]:
        """
        Simulate the outcome of an action.
        Returns a sequence of simulation steps.
        """
        steps_out = []
        current = self._current_state.clone()

        for _ in range(steps):
            # Apply causal rules
            for rule in self._causal_rules:
                if rule['antecedent'] in str(current.variables):
                    if random.random() < rule['probability']:
                        current.variables[rule['consequent']] = True

            # Apply state transition
            state_key = str(sorted(current.variables.items()))
            if state_key in self._state_transition_model:
                transitions = self._state_transition_model[state_key]
                if action.name in transitions:
                    next_key, prob = transitions[action.name]
                    if random.random() < prob:
                        current.variables['last_state'] = next_key

            # Generate outcome description
            outcome = f"Applied {action.name}, state has {len(current.variables)} variables"
            uncertainty = current.uncertainty.get(action.name, 0.0)

            step = SimulationStep(
                state=current.clone(),
                action=action.name,
                outcome=outcome,
                reward=action.expected_reward - action.expected_cost,
                probability=1.0 - uncertainty,
            )
            steps_out.append(step)
            self._simulation_history.append(step)

        return steps_out

    def counterfactual(self, action: Action, alternative_action: Action) -> Dict[str, Any]:
        """
        Compare outcomes of two different actions (counterfactual reasoning).
        """
        # Store original state
        original_state = self._current_state.clone()

        # Simulate original action
        original_outcomes = self.simulate(action, steps=3)

        # Reset and simulate alternative
        self._current_state = original_state.clone()
        alternative_outcomes = self.simulate(alternative_action, steps=3)

        # Reset to original
        self._current_state = original_state

        return {
            'original_action': action.name,
            'original_outcomes': [
                {'step': i, 'outcome': s.outcome, 'reward': s.reward}
                for i, s in enumerate(original_outcomes)
            ],
            'alternative_action': alternative_action.name,
            'alternative_outcomes': [
                {'step': i, 'outcome': s.outcome, 'reward': s.reward}
                for i, s in enumerate(alternative_outcomes)
            ],
            'difference': (
                sum(s.reward for s in original_outcomes) -
                sum(s.reward for s in alternative_outcomes)
            ),
        }

    def to_dict(self) -> Dict:
        # Serialize transitions - convert tuple values to lists for JSON
        serialized_transitions = {}
        for state_key, actions in self._state_transition_model.items():
            serialized_transitions[state_key] = {}
            for action_name, (next_state, prob) in actions.items():
                serialized_transitions[state_key][action_name] = [next_state, prob]
        return {
            'causal_rules': self._causal_rules,
            'transitions': serialized_transitions,
            'state': self._current_state.to_dict() if hasattr(self._current_state, 'to_dict') else {},
        }


# ═══════════════════════════════════════════════════════════════════════
# IV. EXECUTIVE PLANNER
# ═══════════════════════════════════════════════════════════════════════

class ExecutivePlanner:
    """
    Executive planner that optimizes: Reward, Cost, Risk, Confidence, Goal Alignment.
    Uses utility maximization with multi-objective optimization.
    """

    def __init__(self, risk_aversion: float = 0.5, discount_factor: float = 0.9):
        self.risk_aversion = risk_aversion
        self.discount_factor = discount_factor
        self._goals: List[str] = []
        self._goal_weights: Dict[str, float] = {}
        self._constraints: List[str] = []
        self._action_history: List[Action] = []

    def add_goal(self, goal: str, weight: float = 1.0) -> None:
        """Add a goal with its weight."""
        self._goals.append(goal)
        self._goal_weights[goal] = weight

    def add_constraint(self, constraint: str) -> None:
        """Add a planning constraint."""
        self._constraints.append(constraint)

    def evaluate_action(self, action: Action, context: Optional[Dict] = None) -> float:
        """
        Evaluate the utility of an action considering all objectives.
        Utility = Reward - Cost - Risk * risk_aversion + Goal_Alignment
        """
        utility = action.utility(self.risk_aversion)

        # Goal alignment score
        goal_alignment = 0.0
        if self._goals and context:
            for goal in self._goals:
                if goal.lower() in str(context).lower():
                    goal_alignment += self._goal_weights.get(goal, 1.0)

        # Constraint checking
        constraint_penalty = 0.0
        if self._constraints and context:
            for constraint in self._constraints:
                if constraint.lower() in str(context).lower():
                    constraint_penalty += 0.5  # Penalty for actions matching constraints

        return utility + goal_alignment - constraint_penalty

    def plan(self, actions: List[Action], context: Optional[Dict] = None,
             horizon: int = 5) -> Plan:
        """
        Create an optimal plan from available actions.
        Uses greedy look-ahead with utility evaluation.
        """
        if not actions:
            return Plan(goal="", actions=[])

        # Evaluate all actions
        scored_actions = [
            (self.evaluate_action(a, context), a)
            for a in actions
        ]
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # Select top actions within horizon
        selected = []
        total_utility = 0.0
        for score, action in scored_actions[:horizon]:
            if score > 0 or len(selected) == 0:
                selected.append(action)
                total_utility += score

        plan = Plan(
            actions=selected,
            goal=self._goals[0] if self._goals else "unknown",
            total_utility=total_utility,
            constraints=list(self._constraints),
        )

        return plan

    def a_star_search(self, start_state: str, goal_state: str,
                      get_neighbors: Callable[[str], List[Tuple[str, float]]],
                      heuristic: Callable[[str, str], float]) -> Optional[List[str]]:
        """
        A* search for optimal reasoning paths.
        f(n) = g(n) + h(n)
        """
        open_set: List[Tuple[float, float, str, List[str]]] = []
        heapq.heappush(open_set, (0.0, 0.0, start_state, [start_state]))
        closed_set: Set[str] = set()
        g_scores: Dict[str, float] = {start_state: 0.0}

        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current == goal_state:
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor, cost in get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score + cost
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_state)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor,
                                              path + [neighbor]))

        return None

    def mdp_plan(self, states: List[str], actions: List[str],
                 transition_fn: Callable[[str, str], List[Tuple[str, float, float]]],
                 gamma: float = 0.9, iterations: int = 100) -> Dict[str, str]:
        """
        Markov Decision Process value iteration for optimal policy.
        transition_fn(state, action) → [(next_state, probability, reward)]
        """
        # Initialize utilities
        utilities: Dict[str, float] = {s: 0.0 for s in states}
        policy: Dict[str, str] = {s: actions[0] for s in states}

        for _ in range(iterations):
            delta = 0.0
            new_utilities: Dict[str, float] = {}

            for state in states:
                best_utility = float('-inf')
                best_action = actions[0]

                for action in actions:
                    expected = 0.0
                    for next_state, prob, reward in transition_fn(state, action):
                        expected += prob * (reward + gamma * utilities[next_state])

                    if expected > best_utility:
                        best_utility = expected
                        best_action = action

                new_utilities[state] = best_utility
                policy[state] = best_action
                delta = max(delta, abs(utilities[state] - best_utility))

            utilities = new_utilities
            if delta < 1e-6:
                break

        return policy

    def to_dict(self) -> Dict:
        return {
            'goals': self._goals,
            'goal_weights': self._goal_weights,
            'constraints': self._constraints,
            'risk_aversion': self.risk_aversion,
            'discount_factor': self.discount_factor,
        }


# ═══════════════════════════════════════════════════════════════════════
# V. COGNITIVE OS (INTEGRATED)
# ═══════════════════════════════════════════════════════════════════════

class CognitiveOS:
    """
    DUNE-ΩΩΩ: Integrated Cognitive Operating System.
    Combines all cognitive components.
    """

    def __init__(self):
        self.semantic_engine = SemanticResonanceEngine()
        self.memory = MemoryComplex()
        self.world_model = WorldModel()
        self.planner = ExecutivePlanner()

    def perceive(self, input_data: str, input_vector: Optional[List[float]] = None) -> Dict[str, Any]:
        """Process input through the cognitive pipeline."""
        results = {}

        # Semantic processing
        if input_vector:
            field_id, resonance = self.semantic_engine.activate(input_vector)
            if field_id and resonance > 0:
                field_id = self.semantic_engine.learn(input_vector, input_data[:20])
            novelty = self.semantic_engine.detect_novelty(input_vector)
            results['semantic'] = {
                'field_id': field_id,
                'resonance': resonance if field_id else 0,
                'novelty': novelty,
            }

        # Memory processing
        facts = self.memory.query_facts(input_data)
        episodes = self.memory.episodic_memory.recall(input_data)
        results['memory'] = {
            'facts_found': len(facts),
            'episodes_recalled': len(episodes),
        }

        # Store in episodic memory if novel
        if results.get('semantic', {}).get('novelty', 0) > 0.5:
            self.memory.store_experience(input_data, importance=results['semantic']['novelty'])

        return results

    def reason(self, query: str) -> Dict[str, Any]:
        """Reason about a query using all available cognitive resources."""
        results = {
            'query': query,
            'evidence': [],
            'hypotheses': [],
            'conclusion': None,
        }

        # Gather evidence from memory
        facts = self.memory.query_facts(query)
        episodes = self.memory.episodic_memory.recall(query)

        for fact in facts[:5]:
            results['evidence'].append({
                'type': 'fact',
                'content': fact.fact,
                'confidence': fact.confidence,
            })

        for ep in episodes[:3]:
            results['evidence'].append({
                'type': 'episode',
                'content': ep.experience,
                'importance': ep.importance,
            })

        # Generate hypotheses from evidence
        for ev in results['evidence'][:3]:
            results['hypotheses'].append(
                f"Based on: {ev['content'][:50]}..."
            )

        results['conclusion'] = (
            f"Analyzed {len(results['evidence'])} pieces of evidence. "
            f"Confidence: {sum(e.get('confidence', 0.5) for e in results['evidence']) / max(len(results['evidence']), 1):.2f}"
        )

        return results

    def plan(self, goal: str, available_actions: List[Action],
             context: Optional[Dict] = None) -> Plan:
        """Create a plan to achieve a goal."""
        self.planner = ExecutivePlanner()
        self.planner.add_goal(goal)
        return self.planner.plan(available_actions, context)

    def simulate(self, action: Action, steps: int = 3) -> List[SimulationStep]:
        """Simulate an action in the world model."""
        return self.world_model.simulate(action, steps)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the cognitive OS."""
        return {
            'semantic_fields': len(self.semantic_engine.fields),
            'memory': self.memory.get_state_summary(),
            'goals': self.planner._goals,
            'constraints': self.planner._constraints,
        }

    def to_dict(self) -> Dict:
        return {
            'semantic': self.semantic_engine.to_dict(),
            'memory': {
                'debruijn': self.memory.debruijn_memory.to_dict(),
                'episodic': self.memory.episodic_memory.to_dict(),
            },
            'planner': self.planner.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveOS':
        os = cls()
        os.semantic_engine = SemanticResonanceEngine.from_dict(data['semantic'])
        os.memory.debruijn_memory = DeBruijnMemory.from_dict(data['memory']['debruijn'])
        os.memory.episodic_memory = EpisodicMemory.from_dict(data['memory']['episodic'])
        return os


# Convenience alias
DUNE_OMEGA = CognitiveOS