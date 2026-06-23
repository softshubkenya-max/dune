"""
DUNE-LOGIC: Mathematical Reasoning Layer
==========================================
A* Search, Bayesian Inference, Markov Models, TF-IDF Retrieval,
Graph Algorithms, Boolean Logic, Linear Programming, MDPs.
"""

from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
import math
import heapq
import random as _random
from itertools import product as iterproduct


# ═══════════════════════════════════════════════════════════════════════
# I. A* SEARCH
# ═══════════════════════════════════════════════════════════════════════

class AStar:
    """
    A* search algorithm for optimal reasoning paths.
    f(n) = g(n) + h(n)
    """

    @staticmethod
    def search(
        start: str,
        goal: str,
        get_neighbors: Callable[[str], List[Tuple[str, float]]],
        heuristic: Callable[[str, str], float],
        max_states: int = 10000,
    ) -> Optional[Tuple[List[str], float]]:
        """
        Find optimal path from start to goal.
        Returns (path, total_cost) or None if no path found.
        """
        open_set: List[Tuple[float, float, str, List[str]]] = []
        heapq.heappush(open_set, (heuristic(start, goal), 0.0, start, [start]))
        closed_set: Set[str] = set()
        g_scores: Dict[str, float] = {start: 0.0}

        while open_set and len(closed_set) < max_states:
            f_score, g_score, current, path = heapq.heappop(open_set)

            if current == goal:
                return (path, g_score)

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor, cost in get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score + cost
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor,
                                              path + [neighbor]))

        return None

    @staticmethod
    def shortest_path_knowledge_graph(
        kg_nodes: Dict[str, List[Tuple[str, float]]],
        start: str,
        goal: str,
    ) -> Optional[Tuple[List[str], float]]:
        """
        A* search on a knowledge graph.
        kg_nodes: dict mapping node_id -> [(neighbor_id, weight), ...]
        """
        def get_neighbors(node: str) -> List[Tuple[str, float]]:
            return kg_nodes.get(node, [])

        def heuristic(node: str, goal: str) -> float:
            # Simple heuristic: 0 (falls back to Dijkstra)
            return 0.0

        return AStar.search(start, goal, get_neighbors, heuristic)


# ═══════════════════════════════════════════════════════════════════════
# II. BAYESIAN INFERENCE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BayesianNetwork:
    """
    Simple Bayesian network for probabilistic inference.
    Supports prior probabilities and conditional dependencies.
    """
    variables: Dict[str, List[str]] = field(default_factory=dict)  # var -> possible values
    parents: Dict[str, List[str]] = field(default_factory=dict)  # var -> parent vars
    cpts: Dict[str, Dict[Tuple[str, ...], float]] = field(default_factory=dict)  # var -> {assignment_tuple: prob}

    def add_variable(self, name: str, values: List[str],
                     parents: Optional[List[str]] = None) -> None:
        """Add a variable to the network."""
        self.variables[name] = values
        self.parents[name] = parents or []

    def set_cpt(self, variable: str,
                assignments: Dict[Tuple[str, ...], float]) -> None:
        """Set conditional probability table for a variable."""
        self.cpts[variable] = assignments

    def infer(self, query_var: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Perform inference by enumeration.
        Returns probability distribution over query_var given evidence.
        """
        # Get all variables not in evidence or query
        hidden = [v for v in self.variables if v != query_var and v not in evidence]

        # Enumerate all assignments
        result = {}
        for value in self.variables[query_var]:
            prob = 0.0
            # Generate all combinations of hidden variables
            all_hidden_values = [self.variables[v] for v in hidden]
            for hidden_assignment in iterproduct(*all_hidden_values):
                full_assignment = dict(evidence)
                full_assignment[query_var] = value
                for v, val in zip(hidden, hidden_assignment):
                    full_assignment[v] = val
                prob += self._joint_probability(full_assignment)

            result[value] = prob

        # Normalize
        total = sum(result.values())
        if total > 0:
            for k in result:
                result[k] /= total

        return result

    def _joint_probability(self, assignment: Dict[str, str]) -> float:
        """Calculate joint probability P(assignment) using chain rule."""
        prob = 1.0
        for var in self.variables:
            # Get parent assignments
            parent_vals = tuple(assignment[p] for p in self.parents[var])
            # Get probability from CPT
            cpt_key = (assignment[var],) + parent_vals
            var_prob = self.cpts.get(var, {}).get(cpt_key, 0.0)
            prob *= var_prob

            if prob == 0.0:
                break

        return prob


class BayesianReasoner:
    """Bayesian inference for reasoning under uncertainty."""

    def __init__(self):
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[Tuple[str, str], float] = {}
        self.networks: Dict[str, BayesianNetwork] = {}

    def set_prior(self, hypothesis: str, probability: float) -> None:
        self.priors[hypothesis] = probability

    def set_likelihood(self, hypothesis: str, evidence: str, prob: float) -> None:
        self.likelihoods[(hypothesis, evidence)] = prob

    def posterior(self, hypothesis: str, evidence: str) -> float:
        """P(H|E) = P(E|H)P(H) / P(E)"""
        prior = self.priors.get(hypothesis, 0.0)
        if prior == 0:
            return 0.0

        likelihood = self.likelihoods.get((hypothesis, evidence), 0.0)
        if likelihood == 0:
            return 0.0

        # P(E) = sum P(E|Hi)P(Hi)
        evidence_prob = sum(
            self.likelihoods.get((h, evidence), 0.0) * self.priors.get(h, 0.0)
            for h in self.priors
        )

        if evidence_prob == 0:
            return 0.0

        return (likelihood * prior) / evidence_prob

    def rank_hypotheses(self, evidence: str) -> List[Tuple[str, float]]:
        """Rank all hypotheses by posterior probability."""
        results = []
        for h in self.priors:
            p = self.posterior(h, evidence)
            if p > 0:
                results.append((h, p))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def add_network(self, name: str, network: BayesianNetwork) -> None:
        self.networks[name] = network

    def infer_network(self, name: str, query_var: str,
                      evidence: Dict[str, str]) -> Dict[str, float]:
        """Perform inference in a Bayesian network."""
        if name not in self.networks:
            return {}
        return self.networks[name].infer(query_var, evidence)


# ═══════════════════════════════════════════════════════════════════════
# III. MARKOV MODELS
# ═══════════════════════════════════════════════════════════════════════

class MarkovChain:
    """
    Markov chain for sequential transitions.
    P(Xt | Xt-1, ..., Xt-n)
    """

    def __init__(self, order: int = 1):
        self.order = order
        self._transition_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._state_counts: Counter = Counter()

    def add_transition(self, *states: str) -> None:
        """Add a sequence of state transitions."""
        if len(states) < self.order + 1:
            return

        for i in range(len(states) - self.order):
            context = tuple(states[i:i + self.order])
            next_state = states[i + self.order]
            self._transition_counts[context][next_state] += 1
            self._state_counts[context] += 1

    def transition_probability(self, from_states: Tuple[str, ...],
                               to_state: str) -> float:
        """P(to_state | from_states)"""
        if from_states not in self._transition_counts:
            return 0.0
        total = self._state_counts[from_states]
        if total == 0:
            return 0.0
        return self._transition_counts[from_states].get(to_state, 0) / total

    def predict_next(self, *context: str) -> Optional[Tuple[str, float]]:
        """Predict the most likely next state."""
        if len(context) < self.order:
            return None

        key = tuple(context[-self.order:])
        if key not in self._transition_counts:
            return None

        transitions = self._transition_counts[key]
        total = self._state_counts[key]

        if total == 0:
            return None

        best_state, best_count = transitions.most_common(1)[0]
        return (best_state, best_count / total)

    def generate_sequence(self, seed: Tuple[str, ...],
                          length: int = 10) -> List[str]:
        """Generate a sequence of states from the chain."""
        if len(seed) < self.order:
            return list(seed)

        sequence = list(seed)
        for _ in range(length):
            context = tuple(sequence[-self.order:])
            next_state, _ = self.predict_next(*context) or (None, 0)
            if next_state is None:
                break
            sequence.append(next_state)

        return sequence

    def to_dict(self) -> Dict:
        return {
            'order': self.order,
            'transitions': {
                str(k): dict(v) for k, v in self._transition_counts.items()
            },
        }


class HiddenMarkovModel:
    """
    Hidden Markov Model for sequential data with hidden states.
    """

    def __init__(self, n_states: int):
        self.n_states = n_states
        self._transition: List[List[float]] = [[0.0] * n_states for _ in range(n_states)]
        self._emission: List[Dict[str, float]] = [{} for _ in range(n_states)]
        self._initial: List[float] = [1.0 / n_states] * n_states

    def set_transition(self, from_state: int, to_state: int, prob: float) -> None:
        self._transition[from_state][to_state] = prob

    def set_emission(self, state: int, symbol: str, prob: float) -> None:
        self._emission[state][symbol] = prob

    def set_initial(self, probs: List[float]) -> None:
        self._initial = probs

    def viterbi(self, observations: List[str]) -> Tuple[List[int], float]:
        """
        Viterbi algorithm: find most likely hidden state sequence.
        Returns (path, probability).
        """
        n = len(observations)
        if n == 0:
            return ([], 0.0)

        # Initialize
        viterbi_matrix = [[0.0] * self.n_states for _ in range(n)]
        backpointer = [[0] * self.n_states for _ in range(n)]

        for s in range(self.n_states):
            viterbi_matrix[0][s] = (
                self._initial[s] *
                self._emission[s].get(observations[0], 1e-10)
            )

        # Recursion
        for t in range(1, n):
            for s in range(self.n_states):
                max_prob = 0.0
                best_prev = 0
                emission_prob = self._emission[s].get(observations[t], 1e-10)

                for prev_s in range(self.n_states):
                    prob = (viterbi_matrix[t - 1][prev_s] *
                            self._transition[prev_s][s] *
                            emission_prob)
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_s

                viterbi_matrix[t][s] = max_prob
                backpointer[t][s] = best_prev

        # Termination
        best_path_prob = max(viterbi_matrix[n - 1])
        best_last_state = viterbi_matrix[n - 1].index(best_path_prob)

        # Backtrack
        path = [0] * n
        path[n - 1] = best_last_state
        for t in range(n - 2, -1, -1):
            path[t] = backpointer[t + 1][path[t + 1]]

        return (path, best_path_prob)

    def forward(self, observations: List[str]) -> float:
        """
        Forward algorithm: compute P(observations | model).
        Useful for model comparison.
        """
        n = len(observations)
        if n == 0:
            return 0.0

        # Initialize
        alpha = [0.0] * self.n_states
        for s in range(self.n_states):
            alpha[s] = self._initial[s] * self._emission[s].get(observations[0], 1e-10)

        # Recursion
        for t in range(1, n):
            new_alpha = [0.0] * self.n_states
            for s in range(self.n_states):
                emission_prob = self._emission[s].get(observations[t], 1e-10)
                new_alpha[s] = emission_prob * sum(
                    alpha[prev_s] * self._transition[prev_s][s]
                    for prev_s in range(self.n_states)
                )
            alpha = new_alpha

        return sum(alpha)


# ═══════════════════════════════════════════════════════════════════════
# IV. GRAPH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════

class GraphAlgorithms:
    """Graph algorithms for reasoning topology."""

    @staticmethod
    def shortest_path(
        graph: Dict[str, List[Tuple[str, float]]],
        start: str, goal: str,
    ) -> Optional[Tuple[List[str], float]]:
        """Dijkstra's shortest path algorithm."""
        distances: Dict[str, float] = {start: 0.0}
        predecessors: Dict[str, Optional[str]] = {start: None}
        pq: List[Tuple[float, str]] = [(0.0, start)]
        visited: Set[str] = set()

        while pq:
            dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                return (list(reversed(path)), dist)

            for neighbor, weight in graph.get(current, []):
                if neighbor in visited:
                    continue
                new_dist = dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        return None

    @staticmethod
    def connectivity(graph: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Find all connected components."""
        visited: Set[str] = set()
        components: Dict[str, Set[str]] = {}

        def dfs(node: str, component: Set[str]) -> None:
            visited.add(node)
            component.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in graph:
            if node not in visited:
                component: Set[str] = set()
                dfs(node, component)
                for n in component:
                    components[n] = component

        return components

    @staticmethod
    def dependency_order(graph: Dict[str, List[str]]) -> List[str]:
        """
        Topological sort for dependency analysis.
        Graph: node -> list of dependencies (must come before node).
        """
        in_degree: Dict[str, int] = {}
        for node in graph:
            in_degree[node] = len(graph[node])

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

        if len(order) != len(graph):
            return []  # Cycle detected

        return order

    @staticmethod
    def causal_analysis(
        graph: Dict[str, List[Tuple[str, float]]],
        target: str,
    ) -> List[Tuple[str, float]]:
        """
        Find causal factors for a target node using backward traversal.
        Returns (node, influence_score).
        """
        # BFS backwards through the graph
        reverse_graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for src, edges in graph.items():
            for dst, weight in edges:
                reverse_graph[dst].append((src, weight))

        visited: Set[str] = set()
        queue: List[Tuple[str, float]] = [(target, 1.0)]
        causes: List[Tuple[str, float]] = []

        while queue:
            node, influence = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node != target:
                causes.append((node, influence))

            for parent, weight in reverse_graph.get(node, []):
                if parent not in visited:
                    queue.append((parent, influence * weight))

        causes.sort(key=lambda x: x[1], reverse=True)
        return causes


# ═══════════════════════════════════════════════════════════════════════
# V. BOOLEAN LOGIC
# ═══════════════════════════════════════════════════════════════════════

class BooleanLogic:
    """Boolean logic operations for formal correctness."""

    @staticmethod
    def evaluate(expression: str, variables: Dict[str, bool]) -> bool:
        """
        Evaluate a boolean expression.
        Supports: AND, OR, NOT, XOR, IMPLIES, IFF
        Format: "A AND B", "NOT A", "A IMPLIES B", etc.
        """
        expr = expression.strip()
        # Handle parentheses
        while '(' in expr:
            # Find innermost parentheses
            start = expr.rindex('(')
            end = expr.index(')', start)
            inner = expr[start + 1:end]
            result = BooleanLogic.evaluate(inner, variables)
            expr = expr[:start] + str(result) + expr[end + 1:]

        tokens = expr.split()

        # NOT
        while 'NOT' in tokens:
            idx = tokens.index('NOT')
            val = BooleanLogic._to_bool(tokens[idx + 1], variables)
            tokens[idx:idx + 2] = [str(not val)]

        # AND (left to right)
        while 'AND' in tokens:
            idx = tokens.index('AND')
            left = BooleanLogic._to_bool(tokens[idx - 1], variables)
            right = BooleanLogic._to_bool(tokens[idx + 1], variables)
            tokens[idx - 1:idx + 2] = [str(left and right)]

        # OR
        while 'OR' in tokens:
            idx = tokens.index('OR')
            left = BooleanLogic._to_bool(tokens[idx - 1], variables)
            right = BooleanLogic._to_bool(tokens[idx + 1], variables)
            tokens[idx - 1:idx + 2] = [str(left or right)]

        # XOR
        while 'XOR' in tokens:
            idx = tokens.index('XOR')
            left = BooleanLogic._to_bool(tokens[idx - 1], variables)
            right = BooleanLogic._to_bool(tokens[idx + 1], variables)
            tokens[idx - 1:idx + 2] = [str(left != right)]

        # IMPLIES
        while 'IMPLIES' in tokens:
            idx = tokens.index('IMPLIES')
            left = BooleanLogic._to_bool(tokens[idx - 1], variables)
            right = BooleanLogic._to_bool(tokens[idx + 1], variables)
            tokens[idx - 1:idx + 2] = [str((not left) or right)]

        # IFF
        while 'IFF' in tokens:
            idx = tokens.index('IFF')
            left = BooleanLogic._to_bool(tokens[idx - 1], variables)
            right = BooleanLogic._to_bool(tokens[idx + 1], variables)
            tokens[idx - 1:idx + 2] = [str(left == right)]

        return BooleanLogic._to_bool(tokens[0], variables)

    @staticmethod
    def _to_bool(token: str, variables: Dict[str, bool]) -> bool:
        token = token.strip()
        if token == 'True' or token == 'true':
            return True
        if token == 'False' or token == 'false':
            return False
        return variables.get(token, False)

    @staticmethod
    def truth_table(expression: str, variables: List[str]) -> List[Dict[str, Any]]:
        """Generate truth table for an expression."""
        table = []
        for bits in iterproduct([False, True], repeat=len(variables)):
            assignment = dict(zip(variables, bits))
            result = BooleanLogic.evaluate(expression, assignment)
            row = dict(assignment)
            row['result'] = result
            table.append(row)
        return table

    @staticmethod
    def verify(expression: str, variables: List[str]) -> bool:
        """
        Verify if expression is a tautology (always true).
        """
        for bits in iterproduct([False, True], repeat=len(variables)):
            assignment = dict(zip(variables, bits))
            if not BooleanLogic.evaluate(expression, assignment):
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# VI. LINEAR PROGRAMMING (SIMPLEX)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LinearConstraint:
    """A linear constraint: sum(coeffs[i] * vars[i]) {<=, =, >=} rhs"""
    coefficients: List[float]
    variables: List[str]
    relation: str  # <=, =, >=
    rhs: float


class LinearProgram:
    """
    Simple linear programming solver using the simplex method.
    Maximizes c^T x subject to constraints.
    """

    def __init__(self):
        self.objective: List[float] = []
        self.objective_vars: List[str] = []
        self.constraints: List[LinearConstraint] = []
        self._var_index: Dict[str, int] = {}

    def set_objective(self, coefficients: List[float],
                      variables: List[str], maximize: bool = True) -> None:
        self.objective = coefficients
        self.objective_vars = variables
        self._maximize = maximize

    def add_constraint(self, constraint: LinearConstraint) -> None:
        self.constraints.append(constraint)

    def solve(self) -> Optional[Dict[str, float]]:
        """
        Solve the linear program.
        Returns variable assignments or None if infeasible.
        Uses a simplified simplex implementation.
        """
        if not self.objective or not self.constraints:
            return None

        # Collect all variables
        all_vars: List[str] = []
        all_vars.extend(self.objective_vars)
        for c in self.constraints:
            for v in c.variables:
                if v not in all_vars:
                    all_vars.append(v)

        n_vars = len(all_vars)
        self._var_index = {v: i for i, v in enumerate(all_vars)}

        # Simple greedy allocation (production-grade would use full simplex)
        # For now, use a practical heuristic solver

        n_constraints = len(self.constraints)
        # Add slack variables for <= constraints
        n_slack = sum(1 for c in self.constraints if c.relation == '<=')

        m = n_constraints
        n = n_vars + n_slack

        # Initialize tableau
        tableau = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Fill objective (last row)
        for i, v in enumerate(self.objective_vars):
            if v in self._var_index:
                col = self._var_index[v]
                tableau[m][col] = -self.objective[i] if self._maximize else self.objective[i]

        # Fill constraints
        slack_idx = n_vars
        for i, constraint in enumerate(self.constraints):
            for j, v in enumerate(constraint.variables):
                if v in self._var_index:
                    col = self._var_index[v]
                    tableau[i][col] = constraint.coefficients[j]

            if constraint.relation == '<=':
                tableau[i][slack_idx] = 1.0
                slack_idx += 1
            elif constraint.relation == '>=':
                tableau[i][slack_idx] = -1.0
                slack_idx += 1

            tableau[i][n] = constraint.rhs

        # Run simplex
        result = self._simplex(tableau, n_vars, n)
        if result is None:
            return None

        solution = {}
        for i, v in enumerate(all_vars):
            # Check if variable is basic
            basic = True
            for row in result[:-1]:
                if abs(row[i]) > 1e-6 and row[n] > 1e-6:
                    solution[v] = row[n] / row[i] if abs(row[i]) > 1e-6 else 0.0
                    break
            else:
                solution[v] = 0.0

        return solution

    def _simplex(self, tableau: List[List[float]],
                 n_vars: int, n_total: int) -> Optional[List[List[float]]]:
        """Simplex method implementation."""
        m = len(tableau) - 1  # number of constraints
        n = n_total  # total variables including slack

        for iteration in range(1000):
            # Find pivot column (most negative in last row)
            last_row = tableau[m]
            pivot_col = -1
            min_val = 0.0
            for j in range(n):
                if last_row[j] < min_val:
                    min_val = last_row[j]
                    pivot_col = j

            if pivot_col == -1:
                return tableau  # optimal

            # Find pivot row (minimum ratio test)
            pivot_row = -1
            min_ratio = float('inf')
            for i in range(m):
                if tableau[i][pivot_col] > 1e-10:
                    ratio = tableau[i][n] / tableau[i][pivot_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_row = i

            if pivot_row == -1:
                return None  # unbounded

            # Pivot
            pivot_val = tableau[pivot_row][pivot_col]
            for j in range(n + 1):
                tableau[pivot_row][j] /= pivot_val

            for i in range(m + 1):
                if i != pivot_row and abs(tableau[i][pivot_col]) > 1e-10:
                    factor = tableau[i][pivot_col]
                    for j in range(n + 1):
                        tableau[i][j] -= factor * tableau[pivot_row][j]

        return tableau


# ═══════════════════════════════════════════════════════════════════════
# VII. MARKOV DECISION PROCESSES
# ═══════════════════════════════════════════════════════════════════════

class MDP:
    """
    Markov Decision Process for sequential decision-making.
    Solves for optimal policy using value iteration.
    """

    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma
        self.states: Set[str] = set()
        self.actions: Dict[str, List[str]] = defaultdict(list)  # state -> actions
        self.transitions: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
        # (state, action) -> [(next_state, probability, reward)]

    def add_state(self, state: str) -> None:
        self.states.add(state)

    def add_action(self, state: str, action: str) -> None:
        self.actions[state].append(action)

    def add_transition(self, state: str, action: str,
                       next_state: str, probability: float,
                       reward: float) -> None:
        key = (state, action)
        if key not in self.transitions:
            self.transitions[key] = []
        self.transitions[key].append((next_state, probability, reward))
        self.states.add(state)
        self.states.add(next_state)
        if action not in self.actions[state]:
            self.actions[state].append(action)

    def value_iteration(self, iterations: int = 100,
                        tolerance: float = 1e-6) -> Dict[str, float]:
        """Value iteration to find optimal state values."""
        utilities: Dict[str, float] = {s: 0.0 for s in self.states}

        for _ in range(iterations):
            delta = 0.0
            new_utilities: Dict[str, float] = {}

            for state in self.states:
                if not self.actions[state]:
                    new_utilities[state] = 0.0
                    continue

                max_utility = float('-inf')
                for action in self.actions[state]:
                    expected = 0.0
                    key = (state, action)
                    for next_state, prob, reward in self.transitions.get(key, []):
                        expected += prob * (reward + self.gamma * utilities[next_state])
                    max_utility = max(max_utility, expected)

                new_utilities[state] = max_utility if max_utility > float('-inf') else 0.0
                delta = max(delta, abs(new_utilities[state] - utilities[state]))

            utilities = new_utilities
            if delta < tolerance:
                break

        return utilities

    def extract_policy(self, utilities: Dict[str, float]) -> Dict[str, str]:
        """Extract optimal policy from value function."""
        policy: Dict[str, str] = {}
        for state in self.states:
            if not self.actions[state]:
                continue

            best_action = self.actions[state][0]
            best_value = float('-inf')

            for action in self.actions[state]:
                expected = 0.0
                key = (state, action)
                for next_state, prob, reward in self.transitions.get(key, []):
                    expected += prob * (reward + self.gamma * utilities[next_state])
                if expected > best_value:
                    best_value = expected
                    best_action = action

            policy[state] = best_action

        return policy

    def simulate(self, policy: Dict[str, str],
                 start_state: str, steps: int = 10) -> List[Tuple[str, str, float]]:
        """Simulate following a policy."""
        trajectory = []
        current = start_state

        for _ in range(steps):
            if current not in policy:
                break

            action = policy[current]
            key = (current, action)
            transitions = self.transitions.get(key, [])
            if not transitions:
                break

            # Sample from transition probabilities
            r = _random.random()
            cum_prob = 0.0
            chosen_next = None
            chosen_reward = 0.0

            for next_state, prob, reward in transitions:
                cum_prob += prob
                if r <= cum_prob:
                    chosen_next = next_state
                    chosen_reward = reward
                    break

            if chosen_next is None:
                break

            trajectory.append((current, action, chosen_reward))
            current = chosen_next

        return trajectory


# ═══════════════════════════════════════════════════════════════════════
# VIII. DUNE-LOGIC ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class DUNELogic:
    """
    DUNE-Logic: Integrated mathematical reasoning layer.
    Combines all formal reasoning components.
    """

    def __init__(self):
        self.astar = AStar()
        self.bayesian = BayesianReasoner()
        self.markov = MarkovChain(order=2)
        self.hmm: Optional[HiddenMarkovModel] = None
        self.graphs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
        self.boolean = BooleanLogic()
        self.lp = LinearProgram()
        self.mdps: Dict[str, MDP] = {}
        self._knowledge_graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def add_fact(self, subject: str, relation: str, obj: str,
                 weight: float = 1.0) -> None:
        """Add a fact to the reasoning knowledge graph."""
        node_id = f"{subject}_{relation}_{obj}"
        self._knowledge_graph[subject].append((obj, weight))
        self._knowledge_graph[node_id] = [(subject, 1.0), (obj, 1.0)]

    def reason_shortest_path(self, start: str, goal: str) -> Optional[Tuple[List[str], float]]:
        """Find the shortest reasoning path between two concepts."""
        return AStar.shortest_path_knowledge_graph(
            dict(self._knowledge_graph), start, goal
        )

    def reason_graph_path(self, graph_name: str, start: str, goal: str) -> Optional[Tuple[List[str], float]]:
        """Find shortest path in a named graph."""
        if graph_name not in self.graphs:
            return None
        return GraphAlgorithms.shortest_path(self.graphs[graph_name], start, goal)

    def causal_reasoning(self, target: str) -> List[Tuple[str, float]]:
        """Find causal factors for a target."""
        return GraphAlgorithms.causal_analysis(dict(self._knowledge_graph), target)

    def verify_formula(self, expression: str, variables: List[str]) -> bool:
        """Verify if a boolean formula is a tautology."""
        return self.boolean.verify(expression, variables)

    def mdp_solve(self, name: str) -> Optional[Dict[str, str]]:
        """Solve an MDP and return optimal policy."""
        if name not in self.mdps:
            return None
        utilities = self.mdps[name].value_iteration()
        return self.mdps[name].extract_policy(utilities)

    def linear_optimize(self) -> Optional[Dict[str, float]]:
        """Solve the current linear program."""
        return self.lp.solve()

    def to_dict(self) -> Dict:
        return {
            'graph_nodes': len(self._knowledge_graph),
            'markov_order': self.markov.order,
            'mdps': list(self.mdps.keys()),
        }