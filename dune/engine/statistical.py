"""
DUNE-1900: Statistical Learning Engine
========================================
Acquires knowledge continuously from data using classical algorithms.
Pipeline: Internet → Crawler → Corpus → N-grams → Phrase Statistics → TF-IDF → Semantic Graphs → Bayesian Updates
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Iterator
from dataclasses import dataclass, field
import math
import re
import json
from pathlib import Path
from datetime import datetime

from typing import Any
from dune.models.types import (
    NGram, Evidence, KnowledgeAtom, Relation,
    KnowledgeGraph, SemanticMap, SemanticResonanceField
)


class Tokenizer:
    """Simple tokenizer for natural language text."""

    def __init__(self, lower: bool = True, remove_punct: bool = True):
        self.lower = lower
        self.remove_punct = remove_punct
        self._pattern = re.compile(r'\b\w+\b') if remove_punct else re.compile(r'\S+')

    def tokenize(self, text: str) -> List[str]:
        tokens = self._pattern.findall(text)
        if self.lower:
            tokens = [t.lower() for t in tokens]
        return tokens

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class NGramModel:
    """
    N-gram language model with probability estimation.
    Supports up to N-gram order with Laplace smoothing.
    """

    def __init__(self, n: int = 3, smoothing: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.smoothing = smoothing
        self.tokenizer = Tokenizer()
        self._counts: Dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self._total_tokens = 0
        self._vocab: Set[str] = set()
        self._vocab_size = 0

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def update(self, text: str) -> 'NGramModel':
        """Update n-gram counts from text."""
        tokens = self.tokenizer(text)
        self._vocab.update(tokens)
        self._vocab_size = len(self._vocab)
        self._total_tokens += len(tokens)

        for k in range(1, self.n + 1):
            for i in range(len(tokens) - k + 1):
                gram = tuple(tokens[i:i + k])
                self._counts[k][gram] += 1

        return self

    def count(self, gram: Tuple[str, ...]) -> int:
        """Get raw count for an n-gram."""
        k = len(gram)
        if k not in self._counts:
            return 0
        return self._counts[k].get(gram, 0)

    def probability(self, gram: Tuple[str, ...]) -> float:
        """
        Get probability of an n-gram using maximum likelihood estimation
        with Laplace smoothing.
        """
        k = len(gram)
        if k not in self._counts:
            return 0.0

        count = self._counts[k].get(gram, 0)
        # Denominator: total (k-1)-grams or total tokens
        if k == 1:
            total = self._total_tokens
        else:
            prefix = gram[:-1]
            total = self._counts[k - 1].get(prefix, 0)

        smoothed = (count + self.smoothing) / (total + self.smoothing * self.vocab_size) if total > 0 else 0.0
        return smoothed

    def log_probability(self, gram: Tuple[str, ...]) -> float:
        """Get log probability of an n-gram."""
        p = self.probability(gram)
        if p <= 0:
            return -float('inf')
        return math.log2(p)

    def generate(self, seed: Optional[Tuple[str, ...]] = None, max_length: int = 20) -> List[str]:
        """Generate text from the n-gram model."""
        import random
        if seed is None:
            # Start with most common unigram
            if not self._counts[1]:
                return []
            seed = (self._counts[1].most_common(1)[0][0][0],)

        result = list(seed)
        for _ in range(max_length):
            context = tuple(result[-(self.n - 1):]) if self.n > 1 else ()
            candidates = []
            for gram, count in self._counts[self.n].items():
                if gram[:self.n - 1] == context:
                    candidates.extend([gram[-1]] * count)
            if not candidates:
                break
            next_word = random.choice(candidates)
            result.append(next_word)
        return result

    def perplexity(self, text: str) -> float:
        """Calculate perplexity of text under this model."""
        tokens = self.tokenizer(text)
        log_prob_sum = 0.0
        n_grams = 0
        for i in range(len(tokens) - self.n + 1):
            gram = tuple(tokens[i:i + self.n])
            lp = self.log_probability(gram)
            if lp == -float('inf'):
                return float('inf')
            log_prob_sum += lp
            n_grams += 1
        if n_grams == 0:
            return float('inf')
        avg_log_prob = log_prob_sum / n_grams
        return 2 ** (-avg_log_prob)

    def to_dict(self) -> Dict:
        serialized_counts = {}
        for k, c in self._counts.items():
            serialized_counts[str(k)] = {}
            for gram, count in c.items():
                # Use a separator that won't appear in tokens
                serialized_counts[str(k)]['|||'.join(gram)] = count
        return {
            'n': self.n,
            'smoothing': self.smoothing,
            'vocab_size': self.vocab_size,
            'total_tokens': self._total_tokens,
            'counts': serialized_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NGramModel':
        model = cls(n=data['n'], smoothing=data['smoothing'])
        model._total_tokens = data['total_tokens']
        for k, counts in data['counts'].items():
            k = int(k)
            model._counts[k] = Counter()
            for gram_str, count in counts.items():
                gram = tuple(gram_str.split('|||'))
                model._counts[k][gram] = count
                model._vocab.update(gram)
        model._vocab_size = len(model._vocab)
        return model


class TFIDFVectorizer:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
    Provides grounded, transparent factual retrieval.
    """

    def __init__(self):
        self._doc_freq: Counter = Counter()
        self._doc_count = 0
        self._doc_store: Dict[str, List[str]] = {}
        self._idf_cache: Dict[str, float] = {}

    def add_document(self, doc_id: str, text: str, tokenizer: Optional[Tokenizer] = None) -> None:
        """Add a document to the corpus."""
        if tokenizer is None:
            tokenizer = Tokenizer()
        tokens = set(tokenizer(text))
        self._doc_store[doc_id] = tokenizer(text)
        self._doc_count += 1
        for token in tokens:
            self._doc_freq[token] += 1
        self._idf_cache.clear()

    def idf(self, term: str) -> float:
        """Inverse document frequency for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        idf_val = math.log((self._doc_count + 1) / (df + 1)) + 1
        self._idf_cache[term] = idf_val
        return idf_val

    def tf(self, term: str, doc_tokens: List[str]) -> float:
        """Term frequency in a document."""
        count = doc_tokens.count(term)
        return math.log(1 + count) if count > 0 else 0.0

    def tfidf(self, term: str, doc_tokens: List[str]) -> float:
        """TF-IDF score for a term in a document."""
        return self.tf(term, doc_tokens) * self.idf(term)

    def vectorize(self, doc_id: str) -> Dict[str, float]:
        """Get TF-IDF vector for a document."""
        if doc_id not in self._doc_store:
            return {}
        tokens = self._doc_store[doc_id]
        vector = {}
        for token in set(tokens):
            vector[token] = self.tfidf(token, tokens)
        return vector

    def search(self, query: str, top_k: int = 10, tokenizer: Optional[Tokenizer] = None) -> List[Tuple[str, float, str]]:
        """
        Search for documents matching the query.
        Returns list of (doc_id, score, snippet).
        """
        if tokenizer is None:
            tokenizer = Tokenizer()
        query_tokens = tokenizer(query)
        query_vector = {}
        for token in set(query_tokens):
            query_vector[token] = self.idf(token) * math.log(1 + query_tokens.count(token))

        scores = []
        for doc_id, tokens in self._doc_store.items():
            doc_vector = self.vectorize(doc_id)
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > 0:
                snippet = ' '.join(tokens[:50])
                scores.append((doc_id, score, snippet))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve_evidence(self, query: str, top_k: int = 5) -> List[Evidence]:
        """Retrieve evidence for a query."""
        results = self.search(query, top_k=top_k)
        evidence_list = []
        for doc_id, score, snippet in results:
            evidence = Evidence(
                content=snippet,
                source=doc_id,
                relevance_score=score,
            )
            evidence_list.append(evidence)
        return evidence_list

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        dot_product = 0.0
        for term in vec_a:
            if term in vec_b:
                dot_product += vec_a[term] * vec_b[term]
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def to_dict(self) -> Dict:
        return {
            'doc_count': self._doc_count,
            'doc_freq': dict(self._doc_freq),
            'doc_store': {k: v for k, v in self._doc_store.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TFIDFVectorizer':
        tfidf = cls()
        tfidf._doc_count = data['doc_count']
        tfidf._doc_freq = Counter(data['doc_freq'])
        tfidf._doc_store = data['doc_store']
        return tfidf


class BayesianUpdater:
    """
    Bayesian inference engine for belief updates.
    P(H|D) = P(D|H)P(H) / P(D)
    """

    def __init__(self):
        self._priors: Dict[str, float] = {}
        self._likelihoods: Dict[Tuple[str, str], float] = {}
        self._posteriors: Dict[str, float] = {}
        self._evidence_cache: Dict[str, float] = {}

    def set_prior(self, hypothesis: str, probability: float) -> None:
        """Set prior probability for a hypothesis."""
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        self._priors[hypothesis] = probability

    def set_likelihood(self, hypothesis: str, evidence: str, probability: float) -> None:
        """Set P(evidence | hypothesis)."""
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
        self._likelihoods[(hypothesis, evidence)] = probability

    def prior(self, hypothesis: str) -> float:
        """Get prior probability for a hypothesis."""
        return self._priors.get(hypothesis, 0.0)

    def likelihood(self, hypothesis: str, evidence: str) -> float:
        """Get P(evidence | hypothesis)."""
        return self._likelihoods.get((hypothesis, evidence), 0.0)

    def evidence_probability(self, evidence: str, hypotheses: Optional[List[str]] = None) -> float:
        """Calculate P(evidence) = sum(P(evidence|H) * P(H)) over all hypotheses."""
        if evidence in self._evidence_cache:
            return self._evidence_cache[evidence]

        if hypotheses is None:
            hypotheses = list(self._priors.keys())

        prob = 0.0
        for h in hypotheses:
            prob += self.likelihood(h, evidence) * self.prior(h)

        self._evidence_cache[evidence] = prob
        return prob

    def posterior(self, hypothesis: str, evidence: str) -> float:
        """Calculate P(hypothesis | evidence) using Bayes' theorem."""
        prior = self.prior(hypothesis)
        if prior == 0:
            return 0.0

        likelihood = self.likelihood(hypothesis, evidence)
        if likelihood == 0:
            return 0.0

        evidence_prob = self.evidence_probability(evidence)
        if evidence_prob == 0:
            return 0.0

        posterior = (likelihood * prior) / evidence_prob
        self._posteriors[hypothesis] = posterior
        return posterior

    def update(self, hypothesis: str, evidence: str) -> float:
        """Update belief about a hypothesis given evidence. Returns new posterior."""
        posterior = self.posterior(hypothesis, evidence)
        self._priors[hypothesis] = posterior
        self._evidence_cache.clear()
        return posterior

    def rank_hypotheses(self, evidence: str) -> List[Tuple[str, float]]:
        """Rank all hypotheses by their posterior probability given evidence."""
        results = []
        for h in self._priors:
            p = self.posterior(h, evidence)
            results.append((h, p))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def explain(self, hypothesis: str, evidence: str) -> Dict[str, float]:
        """Explain the Bayesian update for a hypothesis."""
        prior = self.prior(hypothesis)
        lik = self.likelihood(hypothesis, evidence)
        ev = self.evidence_probability(evidence)
        post = self.posterior(hypothesis, evidence)

        return {
            'hypothesis': hypothesis,
            'evidence': evidence,
            'prior': prior,
            'likelihood': lik,
            'evidence_probability': ev,
            'posterior': post,
            'bayes_factor': lik / ev if ev > 0 else 0,
        }

    def to_dict(self) -> Dict:
        return {
            'priors': dict(self._priors),
            'likelihoods': {f'{h}|{e}': v for (h, e), v in self._likelihoods.items()},
            'posteriors': dict(self._posteriors),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BayesianUpdater':
        b = cls()
        b._priors = data['priors']
        b._likelihoods = {}
        for key, val in data['likelihoods'].items():
            h, e = key.split('|', 1)
            b._likelihoods[(h, e)] = val
        b._posteriors = data['posteriors']
        return b


class SemanticGraphBuilder:
    """
    Builds semantic graphs from text using co-occurrence and statistical relationships.
    """

    def __init__(self, window_size: int = 5, min_occurrences: int = 2):
        self.window_size = window_size
        self.min_occurrences = min_occurrences
        self.tokenizer = Tokenizer()
        self._cooccurrence: Dict[str, Counter] = defaultdict(Counter)
        self._frequencies: Counter = Counter()
        self._total_docs = 0

    def add_text(self, text: str) -> None:
        """Process text and update co-occurrence statistics."""
        tokens = self.tokenizer(text)
        for i, token in enumerate(tokens):
            self._frequencies[token] += 1
            window_start = max(0, i - self.window_size)
            window_end = min(len(tokens), i + self.window_size + 1)
            for j in range(window_start, window_end):
                if i != j:
                    self._cooccurrence[token][tokens[j]] += 1
        self._total_docs += 1

    def build_graph(self, kg: Optional[KnowledgeGraph] = None) -> KnowledgeGraph:
        """Build a knowledge graph from accumulated statistics."""
        if kg is None:
            kg = KnowledgeGraph()

        # Filter to frequent terms
        frequent_terms = {
            t for t, c in self._frequencies.items()
            if c >= self.min_occurrences
        }

        # Add nodes
        for term in frequent_terms:
            atom = KnowledgeAtom(
                fact=term,
                confidence=min(1.0, self._frequencies[term] / 100),
                source='semantic_graph_builder',
            )
            kg.add_node(atom)

        # Add edges based on pointwise mutual information
        total_cooccurrences = sum(
            sum(c.values()) for c in self._cooccurrence.values()
        )
        total_freq = sum(self._frequencies.values())

        for term_a in frequent_terms:
            for term_b, co_count in self._cooccurrence[term_a].items():
                if term_b not in frequent_terms:
                    continue
                if term_a >= term_b:  # Avoid duplicate edges
                    continue

                # Pointwise Mutual Information
                p_ab = co_count / total_cooccurrences if total_cooccurrences > 0 else 0
                p_a = self._frequencies[term_a] / total_freq if total_freq > 0 else 0
                p_b = self._frequencies[term_b] / total_freq if total_freq > 0 else 0

                if p_a > 0 and p_b > 0 and p_ab > 0:
                    pmi = math.log(p_ab / (p_a * p_b))
                    # Normalize PMI to [0, 1]
                    npmi = max(0, min(1, pmi / (-math.log(p_ab)) if p_ab > 0 else 0))

                    if npmi > 0.1:  # Threshold for meaningful relationships
                        source_id = self._find_node(kg, term_a)
                        target_id = self._find_node(kg, term_b)
                        if source_id and target_id:
                            relation = Relation(
                                source_id=source_id,
                                target_id=target_id,
                                relation_type='co_occurs_with',
                                weight=npmi,
                                confidence=npmi,
                            )
                            kg.add_edge(relation)

        return kg

    def _find_node(self, kg: KnowledgeGraph, term: str) -> Optional[str]:
        """Find a node ID by its fact."""
        for node_id, node in kg.nodes.items():
            if node.fact == term:
                return node_id
        return None

    def to_dict(self) -> Dict:
        return {
            'window_size': self.window_size,
            'min_occurrences': self.min_occurrences,
            'frequencies': dict(self._frequencies),
            'total_docs': self._total_docs,
        }


class DUNE1900:
    """
    DUNE-1900: Main Statistical Learning Engine.
    Integrates all statistical learning components.
    """

    def __init__(self):
        self.ngram_model = NGramModel(n=3)
        self.tfidf = TFIDFVectorizer()
        self.bayesian = BayesianUpdater()
        self.semantic_builder = SemanticGraphBuilder()
        self.knowledge_graph = KnowledgeGraph()
        self.corpus: List[Tuple[str, str, str]] = []  # (source, text, timestamp)
        self.tokenizer = Tokenizer()

    def ingest(self, text: str, source: str = "unknown") -> None:
        """Ingest text into all statistical models."""
        timestamp = datetime.now().isoformat()
        self.corpus.append((source, text, timestamp))
        doc_id = f"doc_{len(self.corpus) - 1}"

        # Update n-gram model
        self.ngram_model.update(text)

        # Update TF-IDF
        self.tfidf.add_document(doc_id, text)

        # Update semantic graph
        self.semantic_builder.add_text(text)

    def build_knowledge_graph(self) -> KnowledgeGraph:
        """Build/rebuild the knowledge graph from all ingested data."""
        self.knowledge_graph = self.semantic_builder.build_graph()
        return self.knowledge_graph

    def retrieve(self, query: str, top_k: int = 10) -> List[Evidence]:
        """Retrieve evidence for a query using TF-IDF."""
        return self.tfidf.retrieve_evidence(query, top_k=top_k)

    def predict_next_words(self, context: str, n_words: int = 5) -> List[Tuple[str, float]]:
        """Predict next words given context."""
        tokens = self.tokenizer(context)
        if len(tokens) < self.ngram_model.n - 1:
            return []

        context_gram = tuple(tokens[-(self.ngram_model.n - 1):])
        candidates = []
        for gram in self.ngram_model._counts[self.ngram_model.n]:
            if gram[:self.ngram_model.n - 1] == context_gram:
                prob = self.ngram_model.probability(gram)
                candidates.append((gram[-1], prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n_words]

    def bayesian_infer(self, hypothesis: str, evidence: str) -> Dict[str, float]:
        """Perform Bayesian inference."""
        return self.bayesian.explain(hypothesis, evidence)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the engine's state."""
        return {
            'corpus_size': len(self.corpus),
            'total_tokens': self.ngram_model._total_tokens,
            'vocab_size': self.ngram_model.vocab_size,
            'kg_nodes': len(self.knowledge_graph.nodes),
            'kg_edges': sum(len(e) for e in self.knowledge_graph.edges.values()),
            'tfidf_docs': self.tfidf._doc_count,
            'n': self.ngram_model.n,
        }

    def to_dict(self) -> Dict:
        return {
            'ngram': self.ngram_model.to_dict(),
            'tfidf': self.tfidf.to_dict(),
            'bayesian': self.bayesian.to_dict(),
            'corpus': self.corpus,
            'stats': self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DUNE1900':
        engine = cls()
        engine.ngram_model = NGramModel.from_dict(data['ngram'])
        engine.tfidf = TFIDFVectorizer.from_dict(data['tfidf'])
        engine.bayesian = BayesianUpdater.from_dict(data['bayesian'])
        engine.corpus = data['corpus']
        # Re-process corpus for semantic builder
        for source, text, ts in engine.corpus:
            engine.semantic_builder.add_text(text)
        engine.build_knowledge_graph()
        return engine


# Convenience alias
DUNE_1900 = DUNE1900