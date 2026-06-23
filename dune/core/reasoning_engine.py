import json
from typing import Dict, Any, List
from dune.core.cognitive import CognitiveOS
from dune.core.concept_encoder import ConceptEncoder
from dune.core.fly_connectome import FlyConnectomeRouter
from dune.core.tribe import TribeFilter

class ReasoningEngine:
    """
    Master Orchestrator for the DUNE-XΩΩΩ-R flow.
    Pipeline: Concept Encoder -> Fly Connectome -> ART/SRF -> TRIBE -> KG/De Bruijn/World Model/Planner -> Decision
    """
    def __init__(self, cognitive_os: CognitiveOS):
        self.cognitive_os = cognitive_os
        self.encoder = ConceptEncoder()
        self.router = FlyConnectomeRouter()
        self.tribe = TribeFilter()
        
    def process(self, llm_json_input: str) -> Dict[str, Any]:
        # 1. Concept Encoder
        encoded = self.encoder.encode(llm_json_input)
        
        # 2. Fly Connectome Router
        circuits = self.router.route(encoded)
        
        # 3. ART / SRF
        field_id, resonance = self.cognitive_os.semantic_engine.activate(encoded["vector"])
        if not field_id:
            field_id = self.cognitive_os.semantic_engine.learn(encoded["vector"], encoded["core_concept"])
            
        # 4. TRIBE + Knowledge Graph
        raw_facts = self.cognitive_os.memory.query_facts(encoded["core_concept"])
        
        # Fallback to general DUNE explanation if memory query is empty
        if not raw_facts:
            # We fetch from the parent orchestrator's explanation pillar theoretically, 
            # but here we just pass the concept string
            filtered_facts = []
        else:
            filtered_facts = self.tribe.filter(raw_facts, encoded["structured_data"].get("goal", ""))
        
        # 5. De Bruijn / World Model / Planner
        goal = encoded["structured_data"].get("goal", "understand")
        self.cognitive_os.planner.add_goal(goal)
        
        # Formulate Decision
        decision_trace = {
            "concept": encoded["core_concept"],
            "domain": encoded["structured_data"].get("domain", "general"),
            "activated_circuits": circuits,
            "art_resonance_score": round(resonance, 4),
            "tribe_filtered_facts": [f.fact for f in filtered_facts],
            "goal": goal,
        }
        
        return decision_trace

    def format_decision_for_llm(self, trace: Dict[str, Any]) -> str:
        return json.dumps(trace, indent=2)
