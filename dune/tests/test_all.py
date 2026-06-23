"""
Comprehensive tests for DUNE-Ω∞-1900-L system.
Tests each pillar individually and the integrated system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
from typing import List, Tuple
from datetime import datetime

from dune.models.types import (
    Evidence, KnowledgeAtom, Relation, KnowledgeGraph,
    MemoryState, Episode, Action, Plan, WorldState,
    SemanticResonanceField, ConceptProfile, RepresentationType, DifficultyLevel
)
from dune.engine.statistical import (
    DUNE1900, NGramModel, TFIDFVectorizer, BayesianUpdater,
    SemanticGraphBuilder, Tokenizer
)
from dune.core.cognitive import (
    CognitiveOS, SemanticResonanceEngine, MemoryComplex,
    DeBruijnMemory, EpisodicMemory, WorldModel, ExecutivePlanner,
    SimulationStep
)
from dune.logic.reasoning import (
    DUNELogic, AStar, BayesianReasoner, MarkovChain,
    HiddenMarkovModel, GraphAlgorithms, BooleanLogic,
    LinearProgram, LinearConstraint, MDP
)
from dune.vi.publishing import (
    DUNEVI, RepresentationEngine, KnowledgeTransformer,
    HumanModel, ContentGenerator
)
from dune import DUNE, create_dune
from dune.llm.openrouter import OpenRouterClient


class TestTypes(unittest.TestCase):
    """Test core data types."""

    def test_evidence(self):
        e = Evidence(content="test", source="src", relevance_score=0.9)
        self.assertEqual(e.content, "test")
        self.assertEqual(e.source, "src")
        self.assertEqual(e.relevance_score, 0.9)
        d = e.to_dict()
        self.assertEqual(d['content'], "test")

    def test_knowledge_atom(self):
        ka = KnowledgeAtom(fact="earth is round", confidence=0.95)
        self.assertEqual(ka.fact, "earth is round")
        self.assertIsNotNone(ka.id)

    def test_relation(self):
        r = Relation(source_id="a", target_id="b", relation_type="causes", weight=0.8)
        self.assertEqual(r.source_id, "a")
        self.assertEqual(r.target_id, "b")

    def test_knowledge_graph(self):
        kg = KnowledgeGraph()
        ka1 = KnowledgeAtom(fact="A")
        ka2 = KnowledgeAtom(fact="B")
        kg.add_node(ka1)
        kg.add_node(ka2)
        r = Relation(ka1.id, ka2.id, "related_to")
        kg.add_edge(r)
        self.assertIn(ka1.id, kg.nodes)
        neighbors = kg.get_neighbors(ka1.id)
        self.assertEqual(len(neighbors), 1)

    def test_action_utility(self):
        a = Action(name="test", expected_reward=10, expected_cost=3, risk=0.5)
        u = a.utility(risk_aversion=0.5)
        self.assertAlmostEqual(u, 10 - 3 - 0.5 * 0.5)


class TestDUNE1900(unittest.TestCase):
    """Test DUNE-1900 Statistical Learning Engine."""

    def setUp(self):
        self.engine = DUNE1900()

    def test_ngram_model(self):
        ng = NGramModel(n=3)
        ng.update("the quick brown fox jumps over the lazy dog")
        self.assertGreater(ng.vocab_size, 0)
        prob = ng.probability(("the", "quick", "brown"))
        self.assertGreater(prob, 0)

    def test_tfidf(self):
        tfidf = TFIDFVectorizer()
        tfidf.add_document("doc1", "machine learning is fascinating")
        tfidf.add_document("doc2", "deep learning is a subset of machine learning")
        results = tfidf.search("machine learning", top_k=2)
        self.assertGreater(len(results), 0)

    def test_openrouter_response_includes_heretic_reference(self):
        client = OpenRouterClient(api_key="dummy-key", model="dummy-model")
        client._call = lambda messages: "Test fluent response."
        response = client.format_response("Test query", "{}")
        self.assertIn("https://github.com/p-e-w/heretic", response)

    def test_bayesian(self):
        bayes = BayesianUpdater()
        bayes.set_prior("disease", 0.01)
        bayes.set_prior("no_disease", 0.99)
        bayes.set_likelihood("disease", "positive_test", 0.9)
        bayes.set_likelihood("no_disease", "positive_test", 0.1)
        post = bayes.posterior("disease", "positive_test")
        self.assertGreater(post, 0)
        self.assertLess(post, 1)
        self.assertAlmostEqual(post, 0.0833, places=3)

    def test_ingest_and_retrieve(self):
        self.engine.ingest("artificial intelligence is transforming technology")
        results = self.engine.retrieve("artificial intelligence", top_k=3)
        self.assertGreater(len(results), 0)

    def test_semantic_graph(self):
        builder = SemanticGraphBuilder()
        builder.add_text("cats and dogs are animals")
        builder.add_text("dogs are pets")
        kg = builder.build_graph()
        self.assertGreater(len(kg.nodes), 0)


class TestCognitiveOS(unittest.TestCase):
    """Test DUNE-ΩΩΩ Cognitive Operating System."""

    def setUp(self):
        self.cos = CognitiveOS()

    def test_semantic_resonance(self):
        engine = SemanticResonanceEngine()
        field = engine.create_field("test", [0.5, 0.5, 0.5])
        self.assertIn(field.id, engine.fields)
        match_id, resonance = engine.activate([0.5, 0.5, 0.5])
        self.assertIsNotNone(match_id)

    def test_debruijn_memory(self):
        mem = DeBruijnMemory(order=2)
        s1 = MemoryState(content={"state": "A"})
        s2 = MemoryState(content={"state": "B"})
        s3 = MemoryState(content={"state": "C"})
        mem.add_state(s1)
        mem.add_state(s2)
        mem.add_state(s3)
        mem.record_transition(s1.id, s2.id, s3.id)
        pred = mem.predict_next(s1.id, s2.id)
        self.assertEqual(pred, s3.id)

    def test_episodic_memory(self):
        mem = EpisodicMemory()
        eid = mem.store("learned about gravity", "understood", importance=0.8)
        self.assertIsNotNone(eid)
        rec = mem.recall("gravity", top_k=3)
        self.assertGreater(len(rec), 0)

    def test_world_model(self):
        wm = WorldModel()
        state = WorldState(variables={"x": 10})
        wm.set_state(state)
        action = Action(name="move", expected_reward=5, expected_cost=1)
        steps = wm.simulate(action, steps=2)
        self.assertGreater(len(steps), 0)

    def test_executive_planner(self):
        planner = ExecutivePlanner()
        planner.add_goal("learn")
        actions = [
            Action(name="read", expected_reward=5, expected_cost=1, risk=0.1),
            Action(name="experiment", expected_reward=8, expected_cost=3, risk=0.4),
        ]
        plan = planner.plan(actions)
        self.assertGreater(len(plan.actions), 0)
        self.assertGreater(plan.total_utility, 0)


class TestDUNELogic(unittest.TestCase):
    """Test DUNE-Logic Mathematical Reasoning Layer."""

    def setUp(self):
        self.logic = DUNELogic()

    def test_astar(self):
        graph = {
            'A': [('B', 1.0), ('C', 4.0)],
            'B': [('C', 2.0), ('D', 5.0)],
            'C': [('D', 1.0)],
            'D': [],
        }
        result = AStar.shortest_path_knowledge_graph(graph, 'A', 'D')
        self.assertIsNotNone(result)
        path, cost = result
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'D')

    def test_bayesian_reasoner(self):
        br = BayesianReasoner()
        br.set_prior("hypothesis_A", 0.5)
        br.set_likelihood("hypothesis_A", "evidence_1", 0.8)
        br.set_prior("hypothesis_B", 0.5)
        br.set_likelihood("hypothesis_B", "evidence_1", 0.2)
        ranked = br.rank_hypotheses("evidence_1")
        self.assertGreater(len(ranked), 0)
        self.assertGreater(ranked[0][1], 0)

    def test_markov_chain(self):
        mc = MarkovChain(order=1)
        mc.add_transition("A", "B", "C", "A", "B")
        pred = mc.predict_next("A")
        self.assertIsNotNone(pred)

    def test_hmm(self):
        hmm = HiddenMarkovModel(n_states=2)
        hmm.set_transition(0, 0, 0.7)
        hmm.set_transition(0, 1, 0.3)
        hmm.set_transition(1, 0, 0.4)
        hmm.set_transition(1, 1, 0.6)
        hmm.set_emission(0, "A", 0.5)
        hmm.set_emission(0, "B", 0.5)
        hmm.set_emission(1, "A", 0.2)
        hmm.set_emission(1, "B", 0.8)
        path, prob = hmm.viterbi(["A", "B", "A"])
        self.assertEqual(len(path), 3)
        self.assertGreater(prob, 0)

    def test_boolean_logic(self):
        result = BooleanLogic.evaluate("A AND B", {"A": True, "B": True})
        self.assertTrue(result)
        result = BooleanLogic.evaluate("A AND B", {"A": True, "B": False})
        self.assertFalse(result)
        is_valid = BooleanLogic.verify("A OR NOT A", ["A"])
        self.assertTrue(is_valid)

    def test_mdp(self):
        mdp = MDP(gamma=0.9)
        mdp.add_transition("s1", "a1", "s2", 0.8, 10.0)
        mdp.add_transition("s1", "a1", "s1", 0.2, 0.0)
        mdp.add_transition("s2", "a1", "s1", 0.5, 5.0)
        mdp.add_transition("s2", "a1", "s2", 0.5, 5.0)
        utilities = mdp.value_iteration()
        self.assertIn("s1", utilities)
        policy = mdp.extract_policy(utilities)
        self.assertIn("s1", policy)

    def test_graph_algorithms(self):
        graph = {
            'A': [('B', 1.0), ('C', 4.0)],
            'B': [('C', 2.0), ('D', 5.0)],
            'C': [('D', 1.0)],
            'D': [],
        }
        result = GraphAlgorithms.shortest_path(graph, 'A', 'D')
        self.assertIsNotNone(result)
        path, cost = result
        self.assertEqual(path, ['A', 'B', 'C', 'D'])
        self.assertAlmostEqual(cost, 4.0)


class TestDUNEVI(unittest.TestCase):
    """Test DUNE-VI Cognitive Publishing Engine."""

    def setUp(self):
        self.vi = DUNEVI()

    def test_register_and_explain(self):
        self.vi.register_concept(
            "test_concept", importance=0.8, difficulty=0.6,
            novelty=0.5, dependencies=["dep1", "dep2"],
            visualizability=0.7, ambiguity=0.3
        )
        explanation = self.vi.explain("test_concept")
        self.assertIn("test_concept", explanation)
        self.assertGreater(len(explanation), 10)

    def test_representation_selection(self):
        engine = RepresentationEngine()
        profile = ConceptProfile(
            name="complex_concept", importance=0.9,
            difficulty=0.8, novelty=0.7,
            dependencies=["a", "b", "c", "d"],
            visualizability=0.3, ambiguity=0.6
        )
        engine.register_concept(profile)
        plan = engine.select_representation("complex_concept")
        self.assertIn('representation', plan)

    def test_human_model(self):
        hm = HumanModel()
        hm.set_level(DifficultyLevel.BEGINNER)
        self.assertEqual(hm.level, DifficultyLevel.BEGINNER)
        hm.record_interaction("test", "story", feedback_score=0.8)
        conf = hm.get_confidence("test")
        self.assertGreater(conf, 0)

    def test_knowledge_transformer(self):
        timeline = KnowledgeTransformer.to_timeline([
            {'timestamp': '2024', 'description': 'Event A'},
            {'timestamp': '2025', 'description': 'Event B'},
        ])
        self.assertIn("Timeline", timeline)

        analogy = KnowledgeTransformer.to_analogy(
            "brain", "neural network",
            [("neurons", "nodes"), ("synapses", "weights")]
        )
        self.assertIn("brain", analogy)


class TestIntegratedDUNE(unittest.TestCase):
    """Test the complete integrated DUNE system."""

    def setUp(self):
        self.dune = create_dune()

    def test_learn(self):
        result = self.dune.learn("The universe is vast and contains many galaxies. Galaxies are vast collections of stars and planets. The universe contains everything.")
        self.assertGreater(result['tokens_added'], 0)
        self.assertGreaterEqual(result['facts_stored'], 0)

    def test_reason(self):
        self.dune.learn("Knowledge is understanding gained through experience")
        result = self.dune.reason("What is knowledge?")
        self.assertIn('evidence', result)
        self.assertIn('conclusion', result)

    def test_plan(self):
        plan = self.dune.plan("research topic")
        self.assertIn('goal', plan)
        self.assertIn('plan', plan)
        self.assertGreater(plan['total_utility'], 0)

    def test_explain(self):
        explanation = self.dune.explain("DUNE")
        self.assertIn("DUNE", explanation)

    def test_explain_with_reasoning(self):
        self.dune.learn("DUNE systems combine learning, reasoning, and communication")
        result = self.dune.explain_with_reasoning("DUNE")
        self.assertIn("DUNE", result)

    def test_feedback_loop(self):
        self.dune.provide_feedback("test_concept", 0.9, "Great explanation")
        self.assertEqual(len(self.dune._feedback_history), 1)

    def test_status(self):
        status = self.dune.status()
        self.assertIn('pillars', status)
        self.assertIn('learning_engine', status)

    def test_pillar_toggle(self):
        self.dune.disable_pillar('logic')
        self.assertFalse(self.dune._enabled_pillars['logic'])
        self.dune.enable_pillar('logic')
        self.assertTrue(self.dune._enabled_pillars['logic'])

    def test_save_load(self):
        self.dune.learn("Test data for save/load")
        self.dune.save_state("/tmp/test_dune_state.json")
        loaded = DUNE.load_state("/tmp/test_dune_state.json")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded._feedback_history), len(self.dune._feedback_history))
        # Cleanup
        import os
        os.remove("/tmp/test_dune_state.json")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_ngram(self):
        ng = NGramModel(n=2)
        prob = ng.probability(("unknown",))
        self.assertEqual(prob, 0.0)

    def test_empty_tfidf_search(self):
        tfidf = TFIDFVectorizer()
        results = tfidf.search("anything")
        self.assertEqual(len(results), 0)

    def test_empty_plan(self):
        planner = ExecutivePlanner()
        plan = planner.plan([])
        self.assertEqual(len(plan.actions), 0)

    def test_unknown_concept_explain(self):
        vi = DUNEVI()
        explanation = vi.explain("nonexistent_concept")
        self.assertIn("not found", explanation)

    def test_empty_episodic_recall(self):
        mem = EpisodicMemory()
        rec = mem.recall("nothing")
        self.assertEqual(len(rec), 0)

    def test_bayesian_no_prior(self):
        bayes = BayesianUpdater()
        post = bayes.posterior("unknown", "evidence")
        self.assertEqual(post, 0.0)

    def test_mdp_no_states(self):
        mdp = MDP()
        utils = mdp.value_iteration()
        self.assertEqual(len(utils), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)