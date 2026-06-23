"""
DUNE-VI: Cognitive Publishing Engine
======================================
Converts cognition into comprehension.
Knowledge → Understanding Model → Representation Selection → Human Understanding
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import textwrap

from typing import TYPE_CHECKING
from dune.models.types import ConceptProfile, RepresentationType, DifficultyLevel

if TYPE_CHECKING:
    from dune.models.types import Evidence


# ═══════════════════════════════════════════════════════════════════════
# I. REPRESENTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class RepresentationEngine:
    """
    Scores concepts on: Importance, Difficulty, Novelty, Dependency,
    Visualizability, Entropy.
    Determines optimal representation for any concept.
    """

    def __init__(self):
        self._concept_profiles: Dict[str, ConceptProfile] = {}
        self._representation_map: Dict[RepresentationType, List[str]] = {
            rt: [] for rt in RepresentationType
        }

    def register_concept(self, profile: ConceptProfile) -> None:
        """Register a concept with its profile."""
        self._concept_profiles[profile.name] = profile

    def profile(self, name: str) -> Optional[ConceptProfile]:
        """Get the profile of a concept."""
        return self._concept_profiles.get(name)

    def _map_representation(self, profile: ConceptProfile) -> RepresentationType:
        """
        Map a concept profile to the optimal representation type.
        Uses entropy-based decision with multi-factor scoring.
        """
        entropy = profile.entropy

        # Very simple concepts: story
        if profile.difficulty < 0.2 and entropy < 0.1:
            return RepresentationType.STORY

        # High visualizability: diagram
        if profile.visualizability > 0.7 and profile.difficulty < 0.6:
            return RepresentationType.DIAGRAM

        # Presentation: high importance, visualizable, multiple dependencies
        if profile.importance > 0.8 and profile.visualizability > 0.6 and profile.dependencies and len(profile.dependencies) >= 3:
            return RepresentationType.PRESENTATION

        # Timeline for time-series or sequential
        if profile.difficulty > 0.3 and profile.visualizability > 0.4:
            return RepresentationType.TIMELINE

        # Process/mechanism: flowchart
        if profile.difficulty > 0.4 and profile.visualizability > 0.5:
            return RepresentationType.FLOWCHART

        # Hierarchy: tree
        if profile.dependencies and len(profile.dependencies) > 2:
            return RepresentationType.TREE

        # Network/complex relations: concept graph
        if profile.entropy > 1.0 or profile.novelty > 0.7:
            return RepresentationType.CONCEPT_GRAPH

        # Comparison
        if profile.ambiguity > 0.6:
            return RepresentationType.COMPARISON

        # Low priority / detail: collapsed
        if profile.importance < 0.3:
            return RepresentationType.COLLAPSED

        # Default: balanced explanation
        return RepresentationType.DIAGRAM

    def select_representation(self, concept_name: str,
                              level: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
                              context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Select the best representation for a concept given audience level.
        Returns a representation plan.
        """
        profile = self._concept_profiles.get(concept_name)
        if not profile:
            return {
                'concept': concept_name,
                'representation': RepresentationType.STORY,
                'confidence': 0.5,
                'reason': 'No profile available, defaulting to story',
            }

        rep_type = self._map_representation(profile)

        # Adjust for audience level
        if level == DifficultyLevel.BEGINNER:
            rep_type = self._adjust_for_beginner(rep_type, profile)
        elif level == DifficultyLevel.EXPERT:
            rep_type = self._adjust_for_expert(rep_type, profile)

        confidence = 1.0 - (profile.entropy / (profile.entropy + 1.0))

        return {
            'concept': concept_name,
            'representation': rep_type,
            'confidence': confidence,
            'entropy': profile.entropy,
            'difficulty': profile.difficulty,
            'novelty': profile.novelty,
            'audience_level': level.name,
            'reason': self._explain_choice(profile, rep_type),
        }

    def _adjust_for_beginner(self, rep_type: RepresentationType,
                              profile: ConceptProfile) -> RepresentationType:
        """Adjust representation for beginner audience."""
        if profile.difficulty > 0.7:
            return RepresentationType.ANALOGY
        if rep_type in (RepresentationType.CONCEPT_GRAPH, RepresentationType.DENSE):
            return RepresentationType.STORY
        return rep_type

    def _adjust_for_expert(self, rep_type: RepresentationType,
                            profile: ConceptProfile) -> RepresentationType:
        """Adjust representation for expert audience."""
        if profile.entropy > 0.5:
            return RepresentationType.FORMAL
        if rep_type in (RepresentationType.STORY, RepresentationType.ANALOGY):
            return RepresentationType.DENSE
        return rep_type

    def _explain_choice(self, profile: ConceptProfile,
                        rep_type: RepresentationType) -> str:
        """Generate explanation for representation choice."""
        reasons = []
        if profile.entropy > 1.0:
            reasons.append(f"high entropy ({profile.entropy:.2f}) requires structured visualization")
        if profile.novelty > 0.7:
            reasons.append("novel concept needs clear framing")
        if profile.visualizability > 0.7:
            reasons.append("highly visualizable concept")
        if profile.difficulty > 0.7:
            reasons.append("difficult concept needs scaffolding")

        return f"Selected {rep_type.name}: {', '.join(reasons)}" if reasons else f"Default representation: {rep_type.name}"

    def get_concept_map(self) -> Dict[str, Dict[str, Any]]:
        """Get the full concept-to-representation mapping."""
        return {
            name: {
                'profile': p.to_dict(),
                'recommended': self._map_representation(p).name,
            }
            for name, p in self._concept_profiles.items()
        }


# ═══════════════════════════════════════════════════════════════════════
# II. KNOWLEDGE TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeTransformer:
    """
    Transforms knowledge structures into human-readable formats.
    Supports multiple representation types.
    """

    @staticmethod
    def to_timeline(events: List[Dict[str, Any]]) -> str:
        """Convert events to a timeline representation."""
        if not events:
            return "No events to display."

        lines = ["╔══════════════════ Timeline ══════════════════╗"]
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', ''))

        for i, event in enumerate(sorted_events):
            ts = event.get('timestamp', 'N/A')
            desc = event.get('description', event.get('content', 'Unknown'))
            lines.append(f"  {ts}")
            lines.append(f"  │  {desc}")
            if i < len(sorted_events) - 1:
                lines.append(f"  ├─────── {event.get('duration', '')}")
            else:
                lines.append(f"  └── (present)")

        lines.append("╚══════════════════════════════════════════════╝")
        return '\n'.join(lines)

    @staticmethod
    def to_flowchart(steps: List[Dict[str, str]]) -> str:
        """Convert process steps to a flowchart."""
        if not steps:
            return "No steps to display."

        lines = ["┌─ FLOWCHART ─────────────────────────────┐"]
        for i, step in enumerate(steps):
            label = step.get('label', step.get('action', f'Step {i + 1}'))
            desc = step.get('description', '')

            lines.append(f"│  [{label}]")
            if desc:
                lines.append(f"│  {desc}")

            if i < len(steps) - 1:
                decision = step.get('decision')
                if decision:
                    lines.append(f"│  ▼  [{decision}]")
                else:
                    lines.append(f"│  ▼")

        lines.append("└────────────────────────────────────────┘")
        return '\n'.join(lines)

    @staticmethod
    def to_tree(hierarchy: Dict[str, Any], indent: int = 0) -> str:
        """Convert hierarchical data to a tree structure."""
        lines = []
        prefix = "  " * indent

        if isinstance(hierarchy, dict):
            for key, value in hierarchy.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}├─ {key}")
                    lines.append(KnowledgeTransformer.to_tree(value, indent + 1))
                else:
                    lines.append(f"{prefix}├─ {key}: {value}")
        elif isinstance(hierarchy, list):
            for item in hierarchy:
                if isinstance(item, (dict, list)):
                    lines.append(KnowledgeTransformer.to_tree(item, indent + 1))
                else:
                    lines.append(f"{prefix}├─ {item}")
        else:
            lines.append(f"{prefix}└─ {hierarchy}")

        return '\n'.join(lines)

    @staticmethod
    def to_concept_graph(concepts: List[Dict[str, Any]],
                         relations: List[Dict[str, Any]]) -> str:
        """Convert concepts and relations to a graph representation."""
        lines = ["╭─ Concept Graph ───────────────────────────╮"]

        # List concepts
        lines.append("│  Nodes:")
        for c in concepts:
            name = c.get('name', c.get('label', 'Unknown'))
            weight = c.get('weight', 1.0)
            lines.append(f"│    • {name} [weight: {weight:.2f}]")

        # List relations
        if relations:
            lines.append("│  Edges:")
            for r in relations:
                src = r.get('source', '?')
                tgt = r.get('target', '?')
                rtype = r.get('type', 'related_to')
                lines.append(f"│    {src} ──{rtype}── {tgt}")

        lines.append("╰────────────────────────────────────────╯")
        return '\n'.join(lines)

    @staticmethod
    def to_cards(items: List[Dict[str, Any]]) -> str:
        """Convert items to card-based comparison view."""
        if not items:
            return "No items to compare."

        lines = []
        for item in items:
            title = item.get('title', item.get('name', 'Item'))
            lines.append(f"┌─ {title} {'─' * 40}┐")
            for key, value in item.items():
                if key not in ('title', 'name'):
                    lines.append(f"│  {key}: {value}")
            lines.append(f"└{'─' * 50}┘")
            lines.append("")

        return '\n'.join(lines)

    @staticmethod
    def to_formal(notation: str, definitions: Optional[Dict[str, str]] = None) -> str:
        """Convert to formal/mathematical notation."""
        lines = ["┌─ Formal Notation ─────────────────────────┐"]
        lines.append(f"│  {notation}")
        if definitions:
            lines.append("│  Where:")
            for symbol, meaning in definitions.items():
                lines.append(f"│    {symbol} = {meaning}")
        lines.append("└────────────────────────────────────────┘")
        return '\n'.join(lines)

    @staticmethod
    def to_story(narrative: str, characters: Optional[List[str]] = None) -> str:
        """Convert to narrative/story format."""
        lines = ["📖 Story Mode"]
        if characters:
            lines.append(f"Characters: {', '.join(characters)}")
        lines.append("")
        lines.append(textwrap.fill(narrative, width=60))
        return '\n'.join(lines)

    @staticmethod
    def to_analogy(source: str, target: str,
                   mappings: List[Tuple[str, str]]) -> str:
        """Create an analogy between two domains."""
        lines = ["💡 Analogy"]
        lines.append(f"  {source}  is like  {target}")
        lines.append("")
        lines.append("  Mapping:")
        for src_attr, tgt_attr in mappings:
            lines.append(f"    • {src_attr}  →  {tgt_attr}")
        return '\n'.join(lines)

    @staticmethod
    def to_presentation(title: str, sections: List[Dict[str, str]]) -> str:
        """Convert to a Reveal.js HTML presentation."""
        slides_html = ""
        for section in sections:
            slides_html += f"""
            <section>
                <h2>{section.get('title', 'Slide')}</h2>
                {section.get('content', '')}
            </section>"""

        html = f"""```presentation
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} Presentation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reset.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/theme/dracula.min.css">
    <style>
        .reveal h1, .reveal h2, .reveal h3 {{ text-transform: none; color: #00d4ff; }}
        .reveal p, .reveal li {{ font-size: 0.8em; color: #e2e8f0; line-height: 1.6; text-align: left; }}
        .reveal ul {{ display: block; padding-left: 2em; }}
        .reveal section {{ padding: 20px; }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            <section>
                <h1>{title}</h1>
                <p style="text-align: center;">Cognitive Knowledge Deck</p>
            </section>{slides_html}
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.js"></script>
    <script>
        Reveal.initialize({{
            controls: true,
            progress: true,
            center: true,
            hash: false,
            transition: 'slide'
        }});
    </script>
</body>
</html>
```"""
        return html


# ═══════════════════════════════════════════════════════════════════════
# III. HUMAN MODEL
# ═══════════════════════════════════════════════════════════════════════

class HumanModel:
    """
    Models the human reader's understanding level.
    Adapts explanations based on audience expertise.
    """

    def __init__(self):
        self._audience_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
        self._concept_mastery: Dict[str, float] = {}
        self._interaction_history: List[Dict] = []
        self._feedback_scores: List[float] = []

    @property
    def level(self) -> DifficultyLevel:
        return self._audience_level

    def set_level(self, level: DifficultyLevel) -> None:
        """Set the audience expertise level."""
        self._audience_level = level

    def record_interaction(self, concept: str, representation: str,
                           feedback_score: Optional[float] = None) -> None:
        """Record an interaction with the user."""
        self._interaction_history.append({
            'concept': concept,
            'representation': representation,
            'feedback': feedback_score,
            'mastery_before': self._concept_mastery.get(concept, 0.0),
        })

        if feedback_score is not None:
            self._feedback_scores.append(feedback_score)
            # Update concept mastery based on feedback
            current = self._concept_mastery.get(concept, 0.0)
            self._concept_mastery[concept] = min(1.0, current + feedback_score * 0.1)

    def get_confidence(self, concept: str) -> float:
        """Get the system's confidence that the user understands this concept."""
        return self._concept_mastery.get(concept, 0.0)

    def assess(self, concept: str, profile: ConceptProfile) -> DifficultyLevel:
        """
        Assess the appropriate difficulty level for explaining a concept
        to this user.
        """
        mastery = self._concept_mastery.get(concept, 0.0)

        # High mastery → expert level
        if mastery > 0.7:
            return DifficultyLevel.EXPERT

        # Low mastery and complex concept → beginner
        if mastery < 0.3 and profile.difficulty > 0.5:
            return DifficultyLevel.BEGINNER

        # Check dependency mastery
        dep_mastery = 0.0
        if profile.dependencies:
            scores = [self._concept_mastery.get(d, 0.0) for d in profile.dependencies]
            dep_mastery = sum(scores) / len(scores) if scores else 0.0

            if dep_mastery < 0.3:
                return DifficultyLevel.BEGINNER

        # Default to current level
        return self._audience_level

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Summarize interaction feedback."""
        if not self._feedback_scores:
            return {'average': 0.0, 'count': 0}

        return {
            'average': sum(self._feedback_scores) / len(self._feedback_scores),
            'count': len(self._feedback_scores),
            'recent': self._feedback_scores[-5:] if len(self._feedback_scores) >= 5 else self._feedback_scores,
            'trend': (
                'improving' if len(self._feedback_scores) >= 2 and
                self._feedback_scores[-1] > self._feedback_scores[-2]
                else 'stable'
            ),
        }

    def to_dict(self) -> Dict:
        return {
            'level': self._audience_level.name,
            'concept_mastery': dict(self._concept_mastery),
            'interactions': len(self._interaction_history),
            'feedback': self.get_feedback_summary(),
        }


# ═══════════════════════════════════════════════════════════════════════
# IV. CONTENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class ContentGenerator:
    """
    Generates human-readable content from knowledge structures.
    Integrates representation engine, knowledge transformer, and human model.
    """

    def __init__(self):
        self.repr_engine = RepresentationEngine()
        self.transformer = KnowledgeTransformer()
        self.human_model = HumanModel()
        self._content_cache: Dict[str, str] = {}

    def explain(self, concept_name: str,
                profile: Optional[ConceptProfile] = None,
                context: Optional[Dict] = None) -> str:
        """
        Generate an explanation for a concept, optimized for the audience.
        """
        if concept_name in self._content_cache and not context:
            return self._content_cache[concept_name]

        if profile is None:
            profile = self.repr_engine.profile(concept_name)
            if profile is None:
                return f"Concept '{concept_name}' not found in knowledge base."

        # Assess audience level
        audience_level = self.human_model.assess(concept_name, profile)

        # Select representation
        plan = self.repr_engine.select_representation(
            concept_name, level=audience_level, context=context
        )

        # Generate content based on representation type
        rep_type = plan['representation']
        content = self._generate_content(rep_type, concept_name, profile, context)

        # Record interaction
        self.human_model.record_interaction(
            concept_name, rep_type.name,
            feedback_score=context.get('feedback') if context else None
        )

        # Cache
        if not context:
            self._content_cache[concept_name] = content

        return content

    def _generate_content(self, rep_type: RepresentationType,
                          concept_name: str,
                          profile: ConceptProfile,
                          context: Optional[Dict]) -> str:
        """Generate content for a specific representation type."""
        header = f"\n{'=' * 60}\n"
        header += f"  {concept_name}"
        header += f"\n{'=' * 60}\n"

        if rep_type == RepresentationType.STORY:
            return header + self._generate_story(concept_name, profile)
        elif rep_type == RepresentationType.ANALOGY:
            return header + self._generate_analogy(concept_name, profile)
        elif rep_type == RepresentationType.DIAGRAM:
            return header + self._generate_diagram(concept_name, profile)
        elif rep_type == RepresentationType.TIMELINE:
            return header + self._generate_timeline(concept_name, profile)
        elif rep_type == RepresentationType.FLOWCHART:
            return header + self._generate_flowchart(concept_name, profile)
        elif rep_type == RepresentationType.TREE:
            return header + self._generate_tree(concept_name, profile)
        elif rep_type == RepresentationType.CONCEPT_GRAPH:
            return header + self._generate_graph(concept_name, profile)
        elif rep_type == RepresentationType.COMPARISON:
            return header + self._generate_comparison(concept_name, profile)
        elif rep_type == RepresentationType.FORMAL:
            return header + self._generate_formal(concept_name, profile)
        elif rep_type == RepresentationType.DENSE:
            return header + self._generate_dense(concept_name, profile)
        elif rep_type == RepresentationType.PRESENTATION:
            return self._generate_presentation(concept_name, profile)
        elif rep_type == RepresentationType.COLLAPSED:
            return header + f"\n[Collapsed view - {concept_name}]"
        else:
            return header + self._generate_balanced(concept_name, profile)

    def _generate_story(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a story-based explanation."""
        lines = [
            f"\nLet me tell you about {concept}.",
            "",
            f"Imagine you're exploring the concept of {concept} for the first time.",
            f"It has a difficulty of {profile.difficulty:.2f} and "
            f"importance of {profile.importance:.2f}.",
            "",
            f"The key ideas to understand are:",
        ]
        for dep in profile.dependencies[:3]:
            lines.append(f"  • First, understand {dep}")
        lines.append("")
        lines.append(f"As you learn more about {concept}, you'll discover "
                     f"how it connects to other ideas in interesting ways.")
        return '\n'.join(lines)

    def _generate_analogy(self, concept: str, profile: ConceptProfile) -> str:
        """Generate an analogy-based explanation."""
        analogies = {
            'neural_network': ('a brain', 'neurons', 'synapses'),
            'algorithm': ('a recipe', 'ingredients', 'steps'),
            'data_structure': ('a filing cabinet', 'folders', 'organization'),
        }
        source, _, _ = analogies.get(concept.lower(), ('a familiar system', 'parts', 'connections'))

        return self.transformer.to_analogy(
            source=source,
            target=concept,
            mappings=[
                (f"components of {source}", f"components of {concept}"),
                ("how they interact", "relational structure"),
                ("the overall pattern", "the core principle"),
            ]
        )

    def _generate_diagram(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a diagram-style explanation."""
        lines = [
            f"\n┌─ {concept} ───────────────────────────────┐",
            f"│                                               │",
            f"│  Importance:  {'█' * int(profile.importance * 20):20s}│",
            f"│  Difficulty:  {'█' * int(profile.difficulty * 20):20s}│",
            f"│  Novelty:     {'█' * int(profile.novelty * 20):20s}│",
            f"│  Entropy:     {profile.entropy:.2f}                      │",
            f"│                                               │",
            f"│  Dependencies:                                 │",
        ]
        for dep in profile.dependencies[:5]:
            lines.append(f"│    ← {dep}")
        lines.append(f"│                                               │")
        lines.append(f"└───────────────────────────────────────────────┘")
        return '\n'.join(lines)

    def _generate_timeline(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a timeline explanation."""
        return self.transformer.to_timeline([
            {'timestamp': 'Origin', 'description': f'{concept} emerges from foundational ideas'},
            {'timestamp': 'Development', 'description': f'Core principles of {concept} are formalized'},
            {'timestamp': 'Current', 'description': f'Modern understanding of {concept}'},
        ])

    def _generate_flowchart(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a flowchart explanation."""
        return self.transformer.to_flowchart([
            {'label': 'Input', 'description': f'Encounter {concept}'},
            {'label': 'Process', 'description': f'Analyze via {profile.difficulty:.0%} difficulty lens'},
            {'label': 'Decision', 'description': 'Novelty check', 'decision': 'Novel or Familiar?'},
            {'label': 'Integrate', 'description': f'Connect with existing knowledge'},
            {'label': 'Output', 'description': f'Understanding of {concept} formed'},
        ])

    def _generate_tree(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a tree explanation."""
        tree_data = {concept: {d: {} for d in profile.dependencies[:5]}}
        return self.transformer.to_tree(tree_data)

    def _generate_graph(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a concept graph."""
        concepts = [{'name': concept, 'weight': 1.0}]
        for dep in profile.dependencies[:5]:
            concepts.append({'name': dep, 'weight': 0.7})

        relations = [{'source': concept, 'target': dep, 'type': 'depends_on'}
                     for dep in profile.dependencies[:5]]

        return self.transformer.to_concept_graph(concepts, relations)

    def _generate_comparison(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a comparison explanation."""
        return self.transformer.to_cards([
            {'title': concept, 'Difficulty': f'{profile.difficulty:.0%}',
             'Novelty': f'{profile.novelty:.0%}', 'Importance': f'{profile.importance:.0%}'},
            {'title': 'Prerequisites', **{d: 'Required' for d in profile.dependencies[:3]}},
        ])

    def _generate_formal(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a formal notation explanation."""
        return self.transformer.to_formal(
            notation=f"∀x ∈ {concept}: P(x) → Q(x)",
            definitions={
                concept: "the concept under analysis",
                "P(x)": "property of the concept",
                "Q(x)": "derived implication",
            }
        )

    def _generate_dense(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a dense, detailed explanation."""
        lines = [
            f"\n{concept} (Detailed Analysis)",
            "=" * 50,
            f"  Entropy: {profile.entropy:.3f}  |  Ambiguity: {profile.ambiguity:.3f}",
            f"  Difficulty: {profile.difficulty:.3f}  |  Novelty: {profile.novelty:.3f}",
            "",
            "Formal Definition:",
            f"  {concept} comprises a set of interrelated components forming",
            "  a coherent conceptual structure with measurable properties.",
            "",
            "Dependencies:",
        ]
        for i, dep in enumerate(profile.dependencies[:8], 1):
            lines.append(f"  {i}. {dep}")
        return '\n'.join(lines)

    def _generate_balanced(self, concept: str, profile: ConceptProfile) -> str:
        """Generate a balanced explanation."""
        lines = [
            f"\n{concept}",
            "-" * 40,
            f"  {concept} is a concept of {profile.difficulty:.0%} difficulty",
            f"  and {profile.importance:.0%} importance.",
            "",
            f"  It relates to: {', '.join(profile.dependencies[:5])}",
            "",
            f"  Key metrics:",
            f"    • Novelty: {profile.novelty:.0%}",
            f"    • Ambiguity: {profile.ambiguity:.0%}",
            f"    • Visualizability: {profile.visualizability:.0%}",
        ]
        return '\n'.join(lines)

    def _generate_presentation(self, concept: str, profile: ConceptProfile) -> str:
        """Generate an HTML presentation."""
        sections = [
            {'title': f"Overview", 'content': f"<p>{concept} is an important cognitive concept.</p><ul><li>Difficulty: {profile.difficulty:.0%}</li><li>Importance: {profile.importance:.0%}</li><li>Visualizability: {profile.visualizability:.0%}</li><li>Novelty: {profile.novelty:.0%}</li></ul>"},
        ]
        
        if profile.dependencies:
            deps_html = "".join([f"<li>{d}</li>" for d in profile.dependencies[:5]])
            sections.append({
                'title': "Dependencies",
                'content': f"<p>Key prerequisites for understanding {concept}:</p><ul>{deps_html}</ul>"
            })
            
        sections.append({
            'title': "Analysis",
            'content': f"<p>With an entropy of {profile.entropy:.2f}, {concept} represents a {'highly structured' if profile.entropy < 0.5 else 'highly complex'} domain requiring careful cognitive processing.</p>"
        })
        
        return self.transformer.to_presentation(concept, sections)

    def get_history(self) -> List[Dict]:
        """Get interaction history."""
        return self.human_model._interaction_history


# ═══════════════════════════════════════════════════════════════════════
# V. DUNE-VI ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class DUNEVI:
    """
    DUNE-VI: Cognitive Publishing Engine.
    Converts knowledge into understanding through adaptive communication.
    """

    def __init__(self):
        self.content_generator = ContentGenerator()
        self.repr_engine = self.content_generator.repr_engine
        self.human_model = self.content_generator.human_model
        self.transformer = self.content_generator.transformer

    def register_concept(self, name: str,
                         importance: float = 0.5,
                         difficulty: float = 0.5,
                         novelty: float = 0.5,
                         dependencies: Optional[List[str]] = None,
                         visualizability: float = 0.5,
                         ambiguity: float = 0.5) -> ConceptProfile:
        """Register a concept for intelligent explanation."""
        profile = ConceptProfile(
            name=name,
            importance=importance,
            difficulty=difficulty,
            novelty=novelty,
            dependencies=dependencies or [],
            visualizability=visualizability,
            ambiguity=ambiguity,
        )
        self.repr_engine.register_concept(profile)
        return profile

    def explain(self, concept: str,
                level: Optional[DifficultyLevel] = None,
                context: Optional[Dict] = None) -> str:
        """Explain a concept to the user."""
        if level is not None:
            self.human_model.set_level(level)
        return self.content_generator.explain(concept, context=context)

    def explain_with_evidence(self, concept: str,
                               evidence_list: List['Evidence']) -> str:
        """Explain a concept supported by evidence."""
        header = f"\n{'=' * 60}\n  {concept} (Evidence-Based)\n{'=' * 60}\n"
        lines = [header]

        for i, ev in enumerate(evidence_list[:5], 1):
            lines.append(f"\n[{i}] Source: {ev.source}")
            lines.append(f"    Relevance: {ev.relevance_score:.2f}")
            lines.append(f"    {ev.content[:200]}")

        return '\n'.join(lines)

    def set_audience(self, level: DifficultyLevel) -> None:
        """Set the audience expertise level."""
        self.human_model.set_level(level)

    def get_feedback(self) -> Dict[str, Any]:
        """Get feedback summary."""
        return self.human_model.get_feedback_summary()

    def provide_feedback(self, concept: str, score: float) -> None:
        """Provide feedback on an explanation."""
        self.human_model.record_interaction(concept, "feedback", score)

    def get_status(self) -> Dict[str, Any]:
        """Get DUNE-VI status."""
        return {
            'registered_concepts': len(self.repr_engine._concept_profiles),
            'audience_level': self.human_model.level.name,
            'interactions': len(self.human_model._interaction_history),
            'cached_explanations': len(self.content_generator._content_cache),
            'feedback': self.human_model.get_feedback_summary(),
        }

    def to_dict(self) -> Dict:
        return {
            'concepts': self.repr_engine.get_concept_map(),
            'human_model': self.human_model.to_dict(),
        }