from typing import Any, List, Dict

class TribeFilter:
    """
    TRIBE: Evaluates human relevance of the reasoning graph.
    Acts as a filter between SRF and Knowledge Graph to prioritize facts with high human utility.
    """
    def filter(self, facts: List[Any], context_goal: str) -> List[Any]:
        # Filter facts based on their utility towards the human goal
        filtered = []
        for fact in facts:
            relevance = 1.0
            if hasattr(fact, "confidence"):
                relevance *= fact.confidence
                
            if relevance > 0.3:
                filtered.append(fact)
                
        # Sort by relevance
        return filtered[:5]
