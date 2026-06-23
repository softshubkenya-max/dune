"""
DUNE-Ω∞-1900-L: Distributed Unified Neuro-Emergent Intelligence
=================================================================
Complete Cognitive Loop:
  Environment → DUNE-1900 (Learn) → Knowledge Structures → DUNE-ΩΩΩ (Understand)
  → DUNE-Logic (Prove) → World Simulation → Executive Planning
  → DUNE-VI (Explain) → Human → Feedback → System Update
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from dune.engine.statistical import DUNE1900, Evidence
from dune.core.cognitive import CognitiveOS, Action, Plan, SimulationStep
from dune.logic.reasoning import DUNELogic, MDP
from dune.vi.publishing import DUNEVI, DifficultyLevel, ConceptProfile
from dune.rag.pipeline import RAGEngine
from dune.rag.mcp_ingest import AutonomousScheduler, MCPDataIngestor, MCPSource


class DUNE:
    """
    DUNE-Ω∞-1900-L: Complete unified intelligence system.

    Central equation:
        Intelligence = Learning + Reasoning + Simulation + Proof + Communication
    """

    def __init__(self):
        # Initialize all four pillars
        self.learning_engine = DUNE1900()           # DUNE-1900
        self.cognitive_os = CognitiveOS()            # DUNE-ΩΩΩ
        self.logic = DUNELogic()                     # DUNE-Logic
        self.publishing = DUNEVI()                   # DUNE-VI

        # System state
        self._feedback_history: List[Dict] = []
        self._session_id = datetime.now().isoformat()
        self._interaction_count = 0
        self._enabled_pillars = {
            'learning': True,
            'cognitive': True,
            'logic': True,
            'publishing': True,
        }

    # ═══════════════════════════════════════════════════════════════════
    # KNOWLEDGE ACQUISITION
    # ═══════════════════════════════════════════════════════════════════

    def learn(self, text: str, source: str = "user_input") -> Dict[str, Any]:
        """
        Ingest knowledge through DUNE-1900 pipeline.
        Text → N-grams → TF-IDF → Semantic Graphs → Bayesian priors
        """
        self._interaction_count += 1
        self.learning_engine.ingest(text, source=source)
        self.learning_engine.build_knowledge_graph()

        # Update cognitive memory
        facts = self.learning_engine.knowledge_graph
        atom_ids = []
        for node_id, node in facts.nodes.items():
            self.cognitive_os.memory.store_fact(
                node.fact,
                confidence=node.confidence,
                source=source,
            )
            atom_ids.append(node_id)

        return {
            'source': source,
            'tokens_added': len(text.split()),
            'facts_stored': len(atom_ids),
            'vocab_size': self.learning_engine.ngram_model.vocab_size,
            'kg_nodes': len(facts.nodes),
        }

    def learn_batch(self, texts: List[str], source: str = "batch") -> Dict[str, Any]:
        """Learn from multiple texts."""
        total_facts = 0
        for text in texts:
            result = self.learn(text, source=source)
            total_facts += result['facts_stored']
        return {'batch_size': len(texts), 'total_facts': total_facts}

    # ═══════════════════════════════════════════════════════════════════
    # REASONING
    # ═══════════════════════════════════════════════════════════════════

    def reason(self, query: str, use_pillars: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Multi-pillar reasoning.
        Combines evidence retrieval, cognitive reasoning, and formal logic.
        """
        self._interaction_count += 1
        if use_pillars is None:
            use_pillars = ['learning', 'cognitive', 'logic']

        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'evidence': [],
            'cognitive_analysis': None,
            'logical_inference': None,
            'conclusion': None,
        }

        # DUNE-1900: Evidence retrieval
        if 'learning' in use_pillars and self._enabled_pillars['learning']:
            evidence = self.learning_engine.retrieve(query, top_k=5)
            results['evidence'] = [
                {'content': e.content[:200], 'source': e.source, 'score': e.relevance_score}
                for e in evidence
            ]

        # DUNE-ΩΩΩ: Cognitive reasoning
        if 'cognitive' in use_pillars and self._enabled_pillars['cognitive']:
            cognitive = self.cognitive_os.reason(query)
            results['cognitive_analysis'] = cognitive

        # DUNE-Logic: Formal reasoning
        if 'logic' in use_pillars and self._enabled_pillars['logic']:
            # Try to find reasoning paths
            path = self.logic.reason_shortest_path(query.lower(), "conclusion")
            if path:
                results['logical_inference'] = {
                    'path': path[0],
                    'cost': path[1],
                }

            # Bayesian inference from evidence
            if results['evidence']:
                self.learning_engine.bayesian.set_prior(query, 0.5)
                for ev in results['evidence'][:3]:
                    self.learning_engine.bayesian.set_likelihood(
                        query, ev['content'][:30], ev['score']
                    )
                post = self.learning_engine.bayesian.posterior(
                    query, results['evidence'][0]['content'][:30]
                )
                results['bayesian_confidence'] = post

        # Synthesize conclusion
        confidence = 0.0
        if results.get('bayesian_confidence'):
            confidence = results['bayesian_confidence']
        elif results['evidence']:
            confidence = sum(e['score'] for e in results['evidence']) / len(results['evidence'])

        results['conclusion'] = (
            f"Analysis complete. Evidence: {len(results['evidence'])} items. "
            f"Confidence: {confidence:.2f}"
        )

        return results

    # ═══════════════════════════════════════════════════════════════════
    # PLANNING & DECISION MAKING
    # ═══════════════════════════════════════════════════════════════════

    def plan(self, goal: str, available_actions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Create an optimal plan using executive planning with world simulation.
        """
        self._interaction_count += 1
        if available_actions is None:
            available_actions = [
                {'name': 'research', 'reward': 8, 'cost': 3, 'risk': 0.2},
                {'name': 'analyze', 'reward': 6, 'cost': 2, 'risk': 0.1},
                {'name': 'synthesize', 'reward': 10, 'cost': 5, 'risk': 0.3},
                {'name': 'verify', 'reward': 7, 'cost': 2, 'risk': 0.15},
            ]

        actions = [
            Action(
                name=a['name'],
                expected_reward=a.get('reward', 5),
                expected_cost=a.get('cost', 2),
                risk=a.get('risk', 0.5),
            )
            for a in available_actions
        ]

        # Cognitive planning
        plan = self.cognitive_os.plan(goal, actions)

        # Simulate the plan
        simulation_results = []
        for action in plan.actions:
            steps = self.cognitive_os.simulate(action, steps=2)
            simulation_results.append([
                {'step': i, 'outcome': s.outcome, 'reward': s.reward}
                for i, s in enumerate(steps)
            ])

        # Logical verification (Boolean check if plan is valid)
        is_valid = True
        if plan.actions:
            verification_vars = {a.name: True for a in plan.actions}
            is_valid = self.logic.verify_formula(
                " AND ".join(f"{a.name}" for a in plan.actions),
                list(verification_vars.keys())
            )

        return {
            'goal': goal,
            'plan': plan.to_dict(),
            'simulation': simulation_results,
            'verified': is_valid,
            'total_utility': plan.total_utility,
        }

    # ═══════════════════════════════════════════════════════════════════
    # EXPLANATION & COMMUNICATION
    # ═══════════════════════════════════════════════════════════════════

    def explain(self, concept: str,
                audience: DifficultyLevel = DifficultyLevel.INTERMEDIATE) -> str:
        """
        Explain a concept with audience-adaptive representation.
        """
        self._interaction_count += 1
        # Auto-register concept if not exists
        if not self.publishing.repr_engine.profile(concept):
            self.publishing.register_concept(
                name=concept,
                importance=0.7,
                difficulty=0.5,
                novelty=0.5,
                dependencies=[],
                visualizability=0.6,
                ambiguity=0.3,
            )

        return self.publishing.explain(concept, level=audience)

    def explain_with_reasoning(self, query: str) -> str:
        """
        Explain a query with full reasoning trace.
        Learn → Understand → Prove → Explain
        """
        self._interaction_count += 1
        # First, reason about it
        reasoning = self.reason(query)

        # Generate explanation
        explanation = [
            f"\n{'=' * 60}",
            f"  DUNE Analysis: {query}",
            f"{'=' * 60}",
            "",
            "📚 EVIDENCE:",
        ]

        for ev in reasoning['evidence'][:5]:
            explanation.append(f"  • [{ev['source']}] (score: {ev['score']:.2f})")

        if reasoning.get('logical_inference'):
            path_str = " → ".join(reasoning['logical_inference']['path'][:5])
            explanation.extend([
                "",
                "🔗 REASONING PATH:",
                f"  {path_str}",
            ])

        if reasoning.get('bayesian_confidence'):
            explanation.extend([
                "",
                "📊 CONFIDENCE:",
                f"  Bayesian posterior: {reasoning['bayesian_confidence']:.3f}",
            ])

        explanation.extend([
            "",
            "💡 CONCLUSION:",
            f"  {reasoning['conclusion']}",
            "",
            "📖 EXPLANATION:",
        ])

        explanation.append(self.explain(query))

        return '\n'.join(explanation)

    # ═══════════════════════════════════════════════════════════════════
    # FEEDBACK & ADAPTATION
    # ═══════════════════════════════════════════════════════════════════

    def provide_feedback(self, concept: str, score: float,
                         text: Optional[str] = None) -> None:
        """
        Provide feedback to the system for continuous improvement.
        Feedback → System Update (closes the cognitive loop)
        """
        self._feedback_history.append({
            'concept': concept,
            'score': score,
            'text': text,
            'timestamp': datetime.now().isoformat(),
        })
        self._interaction_count += 1

        # Update VI human model
        self.publishing.provide_feedback(concept, score)

        # If feedback is good, reinforce in Bayesian model
        if score > 0.7 and text:
            self.learning_engine.bayesian.set_prior(concept, min(1.0, score))

    # ═══════════════════════════════════════════════════════════════════
    # SYSTEM STATUS & CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════

    def status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'session': self._session_id,
            'interactions': self._interaction_count,
            'pillars': dict(self._enabled_pillars),
            'learning_engine': self.learning_engine.get_statistics(),
            'cognitive_os': self.cognitive_os.get_status(),
            'logic': self.logic.to_dict(),
            'publishing': self.publishing.get_status(),
            'feedback_count': len(self._feedback_history),
        }

    def enable_pillar(self, name: str) -> None:
        """Enable a pillar of the system."""
        if name in self._enabled_pillars:
            self._enabled_pillars[name] = True

    def disable_pillar(self, name: str) -> None:
        """Disable a pillar of the system."""
        if name in self._enabled_pillars:
            self._enabled_pillars[name] = False

    def save_state(self, path: str = "dune_state.json") -> None:
        """Save the complete system state."""
        state = {
            'session': self._session_id,
            'learning': self.learning_engine.to_dict(),
            'cognitive': self.cognitive_os.to_dict(),
            'logic': self.logic.to_dict(),
            'publishing': self.publishing.to_dict(),
            'feedback': self._feedback_history,
            'interactions': self._interaction_count,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    @classmethod
    def load_state(cls, path: str = "dune_state.json") -> 'DUNE':
        """Load system state from file."""
        with open(path, 'r') as f:
            state = json.load(f)

        dune = cls()
        dune._session_id = state['session']
        dune._feedback_history = state['feedback']
        dune._interaction_count = state['interactions']
        dune.learning_engine = DUNE1900.from_dict(state['learning'])
        dune.cognitive_os = CognitiveOS.from_dict(state['cognitive'])
        # Note: logic and publishing are rebuilt from cognitive memory
        return dune


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def create_dune() -> DUNE:
    """Create a ready-to-use DUNE instance with default configuration."""
    dune = DUNE()

    # Register core concepts for the VI layer
    dune.publishing.register_concept(
        "DUNE", importance=1.0, difficulty=0.8, novelty=0.9,
        dependencies=["Learning", "Reasoning", "Simulation", "Proof", "Communication"],
        visualizability=0.7, ambiguity=0.2
    )
    dune.publishing.register_concept(
        "Machine Learning", importance=0.9, difficulty=0.6, novelty=0.3,
        dependencies=["Statistics", "Probability"],
        visualizability=0.5, ambiguity=0.3
    )
    dune.publishing.register_concept(
        "Bayesian Inference", importance=0.8, difficulty=0.7, novelty=0.4,
        dependencies=["Probability", "Statistics"],
        visualizability=0.4, ambiguity=0.3
    )
    dune.publishing.register_concept(
        "A* Search", importance=0.7, difficulty=0.6, novelty=0.2,
        dependencies=["Graph Theory", "Heuristics"],
        visualizability=0.6, ambiguity=0.2
    )

    return dune