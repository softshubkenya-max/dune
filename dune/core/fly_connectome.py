from typing import Dict, Any, List

class FlyConnectomeRouter:
    """
    Selects which cognitive circuits to activate based on the Concept Encoder's output.
    Simulates the routing function of the Drosophila connectome.
    """
    def __init__(self):
        self.circuits = {
            "security": ["Threat Circuit", "Authentication Circuit", "Encryption Circuit"],
            "science": ["Epistemology Circuit", "Empirical Circuit", "Validation Circuit"],
            "general": ["Logic Circuit", "Memory Circuit", "Inference Circuit"]
        }
        
    def route(self, concept_data: Dict[str, Any]) -> List[str]:
        domain = concept_data.get("structured_data", {}).get("domain", "").lower()
        
        activated = []
        if "secur" in domain or "threat" in domain or "network" in domain:
            activated.extend(self.circuits["security"])
        elif "scienc" in domain or "quant" in domain or "phys" in domain:
            activated.extend(self.circuits["science"])
        else:
            activated.extend(self.circuits["general"])
            
        return activated
