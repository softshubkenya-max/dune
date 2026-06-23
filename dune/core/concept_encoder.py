import json
from typing import Dict, Any, List

class ConceptEncoder:
    """
    Converts structured JSON concepts from the Input LLM into vector/symbolic formats
    for ingestion into the Semantic Resonance Fields (ART).
    """
    def encode(self, concept_json: str) -> Dict[str, Any]:
        try:
            # Clean possible markdown formatting
            clean_json = concept_json.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            data = json.loads(clean_json)
        except Exception:
            # Fallback if not pure JSON
            data = {"object": concept_json, "domain": "general", "goal": "explain", "environment": "unknown"}
            
        # Create a deterministic pseudo-embedding based on the text hash for ART resonance
        vector = self._pseudo_embed(str(data.get("object", "")))
        
        return {
            "structured_data": data,
            "vector": vector,
            "core_concept": str(data.get("object", "unknown"))
        }
        
    def _pseudo_embed(self, text: str) -> List[float]:
        # Simple pseudo-embedding for demonstration in the DUNE ART engine
        if not text: return [0.0]*10
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [float(b)/255.0 for b in h[:10]]
