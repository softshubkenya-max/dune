"""
DUNE-Ω∞-1900-L Web UI Server
================================
Self-contained HTTP server with no external dependencies.
Serves the DUNE cognitive AI system via a web interface.
"""

import http.server
import json
import os
import sys
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from streaming import StreamingResponse, IncrementalResponse

from dune import DUNE, create_dune
from dune.rag.mcp_ingest import AutonomousScheduler
from dune.llm.openrouter import OpenRouterClient
from dune.core.reasoning_engine import ReasoningEngine
from dune.db.chat_store import ChatStore

# ═══════════════════════════════════════════════════════════════════════
# Global DUNE Instance & Scheduler
# ═══════════════════════════════════════════════════════════════════════

dune = create_dune()

llm_client = OpenRouterClient(
    api_key=os.environ.get('OPENROUTER_API_KEY', '')
)

# Initialize DUNE-XΩΩΩ-R Reasoning Engine
reasoning_engine = ReasoningEngine(cognitive_os=dune.cognitive_os)

# Initialize Database for Sessions
chat_store = ChatStore()

scheduler = AutonomousScheduler(dune_instance=dune)
# arXiv Sources
scheduler.ingestor.add_api_source("arXiv_quantum", "http://export.arxiv.org/api/query?search_query=all:quantum&start=0&max_results=5")
scheduler.ingestor.add_api_source("arXiv_ai", "http://export.arxiv.org/api/query?search_query=all:artificial+intelligence&start=0&max_results=5")
scheduler.ingestor.add_api_source("arXiv_bio", "http://export.arxiv.org/api/query?search_query=all:biology&start=0&max_results=5")
scheduler.ingestor.add_api_source("arXiv_math", "http://export.arxiv.org/api/query?search_query=all:mathematics&start=0&max_results=5")
scheduler.ingestor.add_api_source("arXiv_physics", "http://export.arxiv.org/api/query?search_query=all:physics&start=0&max_results=5")

# Open Scientific Databases
scheduler.ingestor.add_api_source("OpenAlex_AI", "https://api.openalex.org/works?search=artificial+intelligence&per-page=5")
scheduler.ingestor.add_api_source("OpenAlex_Physics", "https://api.openalex.org/works?search=physics&per-page=5")
scheduler.ingestor.add_api_source("Crossref_Science", "https://api.crossref.org/works?query=science&select=title,abstract&rows=5")
scheduler.ingestor.add_api_source("PLOS_Science", "http://api.plos.org/search?q=title:science&rows=5")
scheduler.ingestor.add_api_source("PubMed_Health", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=health&retmode=json&retmax=5")

scheduler.start(daemon=True)

# Pre-load with some initial knowledge
dune.learn("DUNE is a Distributed Unified Neuro-Emergent Intelligence system that implements a complete cognitive loop of learning, reasoning, simulation, planning, and explanation.")
dune.learn("Bayesian inference updates prior beliefs with new evidence using Bayes' theorem: P(H|D) = P(D|H)P(H) / P(D)")
dune.learn("A* search finds optimal paths using f(n) = g(n) + h(n) where g is cost so far and h is heuristic estimate.")
dune.learn("Semantic Resonance Fields form meaning through pattern matching and adaptive resonance theory.")
dune.learn("Markov Decision Processes model sequential decision-making with state transitions and rewards.")

# Register core concepts for VI
dune.publishing.register_concept(
    "DUNE", importance=1.0, difficulty=0.8, novelty=0.9,
    dependencies=["Learning", "Reasoning", "Simulation", "Proof", "Communication"],
    visualizability=0.7, ambiguity=0.2
)
dune.publishing.register_concept(
    "Bayesian Inference", importance=0.9, difficulty=0.7, novelty=0.5,
    dependencies=["Probability Theory", "Statistics"],
    visualizability=0.5, ambiguity=0.3
)
dune.publishing.register_concept(
    "A* Search", importance=0.8, difficulty=0.6, novelty=0.4,
    dependencies=["Graph Theory", "Heuristics"],
    visualizability=0.7, ambiguity=0.2
)
dune.publishing.register_concept(
    "Semantic Resonance", importance=0.8, difficulty=0.7, novelty=0.6,
    dependencies=["Pattern Recognition", "Adaptive Resonance Theory"],
    visualizability=0.5, ambiguity=0.4
)
dune.publishing.register_concept(
    "Markov Decision Process", importance=0.9, difficulty=0.7, novelty=0.5,
    dependencies=["Probability", "Decision Theory"],
    visualizability=0.6, ambiguity=0.3
)


# ═══════════════════════════════════════════════════════════════════════
# API Handler
# ═══════════════════════════════════════════════════════════════════════

class DUNEAPIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the DUNE web UI."""

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode('utf-8'))
    
    def _send_stream(self, generator, status: int = 200) -> None:
        """Send streaming response with Server-Sent Events."""
        self.send_response(status)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        
        try:
            for chunk in generator:
                self.wfile.write(chunk.encode('utf-8'))
                self.wfile.flush()
        except Exception as e:
            error_event = StreamingResponse.stream_sse_event("error", {
                "error": str(e)
            })
            self.wfile.write(error_event.encode('utf-8'))

    def _send_html(self, content: str, status: int = 200) -> None:
        """Send an HTML response."""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def _read_body(self) -> Dict:
        """Read and parse JSON request body."""
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode('utf-8'))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Serve static files
        if path == '/' or path == '/client' or path == '/client.html':
            self._serve_client()
        elif path == '/dev' or path == '/dev.html':
            self._serve_dev()
        elif path == '/av' or path == '/av.html':
            self._serve_av()
        elif path == '/api/status':
            self._handle_status()
        elif path == '/api/concepts':
            self._handle_concepts()
        elif path == '/api/feedback':
            self._handle_get_feedback()
        elif path == '/api/mcp/status':
            self._handle_mcp_status()
        elif path == '/api/config/openrouter':
            self._handle_get_openrouter_config()
        elif path == '/api/models':
            self._handle_all_models()
        elif path == '/api/sessions':
            self._handle_get_sessions()
        elif path.startswith('/api/sessions/') and path.endswith('/messages'):
            self._handle_get_session_messages(path)
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # Check for streaming mode
        stream = 'stream' in query

        try:
            body = self._read_body()

            if path == '/api/learn':
                if stream:
                    self._handle_learn_stream(body)
                else:
                    self._handle_learn(body)
            elif path == '/api/reason':
                if stream:
                    self._handle_reason_stream(body)
                else:
                    self._handle_reason(body)
            elif path == '/api/explain':
                if stream:
                    self._handle_explain_stream(body)
                else:
                    self._handle_explain(body)
            elif path == '/api/plan':
                if stream:
                    self._handle_plan_stream(body)
                else:
                    self._handle_plan(body)
            elif path == '/api/feedback':
                self._handle_feedback(body)
            elif path == '/api/explain-with-reasoning':
                if stream:
                    self._handle_explain_with_reasoning_stream(body)
                else:
                    self._handle_explain_with_reasoning(body)
            elif path == '/api/mcp/source':
                self._handle_mcp_source(body)
            elif path == '/api/config/openrouter':
                self._handle_set_openrouter_config(body)
            elif path == '/api/chat':
                if stream:
                    self._handle_chat_stream(body)
                else:
                    self._handle_chat(body)
            elif path == '/api/sessions':
                self._handle_create_session(body)
            else:
                self._send_json({'error': 'Not found'}, 404)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def do_DELETE(self) -> None:
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            if path.startswith('/api/sessions/'):
                self._handle_delete_session(path)
            else:
                self._send_json({'error': 'Not found'}, 404)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    # ─── Static File Serving ───

    def _serve_client(self) -> None:
        """Serve the client HTML page."""
        html_path = os.path.join(os.path.dirname(__file__), 'client.html')
        try:
            with open(html_path, 'r') as f:
                content = f.read()
            self._send_html(content)
        except FileNotFoundError:
            self._send_html("<h1>DUNE UI</h1><p>client.html not found</p>", 404)

    def _serve_dev(self) -> None:
        """Serve the developer HTML page."""
        html_path = os.path.join(os.path.dirname(__file__), 'dev.html')
        try:
            with open(html_path, 'r') as f:
                content = f.read()
            self._send_html(content)
        except FileNotFoundError:
            self._send_html("<h1>DUNE UI</h1><p>dev.html not found</p>", 404)

    def _serve_av(self) -> None:
        """Serve the Audio-Visual chatbot HTML page."""
        html_path = os.path.join(os.path.dirname(__file__), 'av.html')
        try:
            with open(html_path, 'r') as f:
                content = f.read()
            self._send_html(content)
        except FileNotFoundError:
            self._send_html("<h1>DUNE UI</h1><p>av.html not found</p>", 404)

    # ─── API Endpoints ───

    def _handle_status(self) -> None:
        """GET /api/status - System status."""
        self._send_json(dune.status())

    def _handle_concepts(self) -> None:
        """GET /api/concepts - Registered concepts."""
        concepts = dune.publishing.repr_engine.get_concept_map()
        self._send_json(concepts)

    def _handle_learn(self, body: Dict) -> None:
        """POST /api/learn - Learn from text."""
        text = body.get('text', '')
        source = body.get('source', 'web_ui')
        if not text:
            self._send_json({'error': 'No text provided'}, 400)
            return
        result = dune.learn(text, source=source)
        self._send_json(result)

    def _handle_reason(self, body: Dict) -> None:
        """POST /api/reason - Reason about a query."""
        query = body.get('query', '')
        if not query:
            self._send_json({'error': 'No query provided'}, 400)
            return
        result = dune.reason(query)
        self._send_json(result)

    def _handle_explain(self, body: Dict) -> None:
        """POST /api/explain - Explain a concept."""
        concept = body.get('concept', '')
        audience = body.get('audience', 'INTERMEDIATE')
        if not concept:
            self._send_json({'error': 'No concept provided'}, 400)
            return

        from dune.models.types import DifficultyLevel
        level_map = {
            'BEGINNER': DifficultyLevel.BEGINNER,
            'INTERMEDIATE': DifficultyLevel.INTERMEDIATE,
            'EXPERT': DifficultyLevel.EXPERT,
        }
        level = level_map.get(audience.upper(), DifficultyLevel.INTERMEDIATE)

        explanation = dune.explain(concept, audience=level)
        self._send_json({
            'concept': concept,
            'audience': audience,
            'explanation': explanation,
        })

    def _handle_plan(self, body: Dict) -> None:
        """POST /api/plan - Create a plan."""
        goal = body.get('goal', '')
        if not goal:
            self._send_json({'error': 'No goal provided'}, 400)
            return
        result = dune.plan(goal)
        self._send_json(result)

    def _handle_feedback(self, body: Dict) -> None:
        """POST /api/feedback - Provide feedback."""
        concept = body.get('concept', '')
        score = body.get('score', 0.5)
        text = body.get('text', '')
        if not concept:
            self._send_json({'error': 'No concept provided'}, 400)
            return
        dune.provide_feedback(concept, float(score), text)
        self._send_json({'status': 'ok', 'concept': concept, 'score': score})

    def _handle_get_feedback(self) -> None:
        """GET /api/feedback - Get feedback summary."""
        self._send_json(dune.publishing.get_feedback())

    def _handle_explain_with_reasoning(self, body: Dict) -> None:
        """POST /api/explain-with-reasoning - Full reasoning trace."""
        query = body.get('query', '')
        if not query:
            self._send_json({'error': 'No query provided'}, 400)
            return
        explanation = dune.explain_with_reasoning(query)
        reasoning = dune.reason(query)
        self._send_json({
            'query': query,
            'reasoning': reasoning,
            'explanation': explanation,
        })

    def _handle_mcp_status(self) -> None:
        self._send_json(scheduler.get_status())

    def _handle_mcp_source(self, body: Dict) -> None:
        name = body.get('name')
        stype = body.get('source_type')
        uri = body.get('uri')
        if not all([name, stype, uri]):
            self._send_json({'error': 'Missing fields'}, 400)
            return
        if stype == 'api':
            scheduler.ingestor.add_api_source(name, uri)
        else:
            scheduler.ingestor.add_url_source(name, uri)
        self._send_json({'status': 'ok'})

    def _handle_get_openrouter_config(self) -> None:
        self._send_json({
            'api_key': llm_client.api_key,
            'model': llm_client.model
        })

    def _handle_set_openrouter_config(self, body: Dict) -> None:
        if 'api_key' in body:
            llm_client.api_key = body['api_key']
        if 'model' in body:
            llm_client.model = body['model']
        self._send_json({'status': 'ok'})

    def _handle_all_models(self) -> None:
        """GET /api/models - List all models from OpenRouter."""
        llm_client._models_cache = None  # force refresh
        models = llm_client.fetch_all_models()
        self._send_json({'models': models, 'current': llm_client.model})

    def _handle_chat(self, body: Dict) -> None:
        """POST /api/chat - Fluent chat with LLM + DUNE"""
        query = body.get('query', '')
        session_id = body.get('session_id')
        
        if not query:
            self._send_json({'error': 'No query provided'}, 400)
            return
            
        if session_id:
            chat_store.add_message(session_id, 'user', query)
            
        # 1. LLM extracts core query as JSON
        core_query_json = llm_client.extract_query(query)
        
        # 2. DUNE-XΩΩΩ-R Reasoning Pipeline
        # Concept Encoder -> Fly Connectome -> ART/SRF -> TRIBE -> KG/Planner
        decision_trace = reasoning_engine.process(core_query_json)
        
        # We always ask DUNE to explain the core concept to get robust facts
        explanation = dune.explain_with_reasoning(decision_trace.get("concept", query))
        decision_trace["dune_vi_explanation"] = explanation
            
        formatted_trace = reasoning_engine.format_decision_for_llm(decision_trace)
        
        # 3. LLM formats output
        fluent_response = llm_client.format_response(query, formatted_trace)
        
        if session_id:
            chat_store.add_message(session_id, 'bot', fluent_response)
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(fluent_response.encode('utf-8'))

    def _handle_get_sessions(self) -> None:
        self._send_json(chat_store.get_sessions())
        
    def _handle_get_session_messages(self, path: str) -> None:
        session_id = path.split('/')[3]
        self._send_json(chat_store.get_messages(session_id))
        
    def _handle_create_session(self, body: Dict) -> None:
        title = body.get('title', 'New Chat')
        session = chat_store.create_session(title)
        self._send_json(session)
        
    def _handle_delete_session(self, path: str) -> None:
        session_id = path.split('/')[3]
        success = chat_store.delete_session(session_id)
        if success:
            self._send_json({'status': 'ok'})
        else:
            self._send_json({'error': 'Not found or failed to delete'}, 404)
    
    # ─── Streaming Handlers ───
    
    def _handle_learn_stream(self, body: Dict) -> None:
        """POST /api/learn?stream - Stream learning results."""
        text = body.get('text', '')
        source = body.get('source', 'web_ui')
        if not text:
            self._send_json({'error': 'No text provided'}, 400)
            return
        
        def learn_generator():
            yield StreamingResponse.stream_sse_event("start", {"operation": "learn"})
            result = dune.learn(text, source=source)
            
            # Stream result pieces
            if isinstance(result, dict):
                for key, value in result.items():
                    yield StreamingResponse.stream_sse_event("data", {
                        "key": key,
                        "value": value
                    })
            else:
                yield StreamingResponse.stream_sse_event("chunk", {
                    "content": str(result)
                })
            
            yield StreamingResponse.stream_sse_event("complete", {"status": "ok"})
        
        self._send_stream(learn_generator())

    def _handle_chat_stream(self, body: Dict) -> None:
        """POST /api/chat?stream - Stream chat response."""
        query = body.get('query', '')
        session_id = body.get('session_id')
        
        if not query:
            self._send_json({'error': 'No query provided'}, 400)
            return
        
        if session_id:
            chat_store.add_message(session_id, 'user', query)
        
        def chat_generator():
            yield StreamingResponse.stream_sse_event("start", {"operation": "chat"})
            try:
                # 1. LLM extracts core query as JSON
                core_query_json = llm_client.extract_query(query)
                yield StreamingResponse.stream_sse_event("progress", {"stage": "extracted_query"})
                
                # 2. Reasoning pipeline
                decision_trace = reasoning_engine.process(core_query_json)
                yield StreamingResponse.stream_sse_event("progress", {"stage": "reasoning"})
                
                # Explanation
                explanation = dune.explain_with_reasoning(decision_trace.get("concept", query))
                decision_trace["dune_vi_explanation"] = explanation
                yield StreamingResponse.stream_sse_event("progress", {"stage": "explanation"})
                
                formatted_trace = reasoning_engine.format_decision_for_llm(decision_trace)
                
                # 3. LLM formats output - stream word by word
                fluent_response = llm_client.format_response(query, formatted_trace)
                words = str(fluent_response).split()
                for i, word in enumerate(words):
                    yield StreamingResponse.stream_sse_event("chunk", {"content": word, "index": i})
                
                if session_id:
                    chat_store.add_message(session_id, 'bot', fluent_response)
                
                yield StreamingResponse.stream_sse_event("complete", {"status": "ok"})
            except Exception as e:
                yield StreamingResponse.stream_sse_event("error", {"error": str(e)})
        
        self._send_stream(chat_generator())
    
    def _handle_reason_stream(self, body: Dict) -> None:
        """POST /api/reason?stream - Stream reasoning process."""
        query = body.get('query', '')
        if not query:
            self._send_json({'error': 'No query provided'}, 400)
            return
        
        def reason_generator():
            yield StreamingResponse.stream_sse_event("start", {"operation": "reason"})
            result = dune.reason(query)
            
            # Stream reasoning steps
            words = str(result).split()
            for i, word in enumerate(words):
                yield StreamingResponse.stream_sse_event("chunk", {
                    "content": word,
                    "index": i,
                    "total": len(words)
                })
            
            yield StreamingResponse.stream_sse_event("complete", {"status": "ok"})
        
        self._send_stream(reason_generator())
    
    def _handle_explain_stream(self, body: Dict) -> None:
        """POST /api/explain?stream - Stream explanation."""
        concept = body.get('concept', '')
        audience = body.get('audience', 'INTERMEDIATE')
        if not concept:
            self._send_json({'error': 'No concept provided'}, 400)
            return

        from dune.models.types import DifficultyLevel
        level_map = {
            'BEGINNER': DifficultyLevel.BEGINNER,
            'INTERMEDIATE': DifficultyLevel.INTERMEDIATE,
            'EXPERT': DifficultyLevel.EXPERT,
        }
        level = level_map.get(audience.upper(), DifficultyLevel.INTERMEDIATE)

        def explain_generator():
            yield StreamingResponse.stream_sse_event("start", {
                "operation": "explain",
                "concept": concept
            })
            
            explanation = dune.explain(concept, audience=level)
            
            # Stream explanation sentence by sentence
            sentences = str(explanation).split('. ')
            for i, sentence in enumerate(sentences):
                yield StreamingResponse.stream_sse_event("chunk", {
                    "content": sentence + ('. ' if i < len(sentences) - 1 else ''),
                    "index": i,
                    "total": len(sentences)
                })
            
            yield StreamingResponse.stream_sse_event("complete", {
                "status": "ok",
                "concept": concept,
                "audience": audience
            })
        
        self._send_stream(explain_generator())
    
    def _handle_plan_stream(self, body: Dict) -> None:
        """POST /api/plan?stream - Stream planning process."""
        goal = body.get('goal', '')
        if not goal:
            self._send_json({'error': 'No goal provided'}, 400)
            return
        
        def plan_generator():
            yield StreamingResponse.stream_sse_event("start", {
                "operation": "plan",
                "goal": goal
            })
            
            result = dune.plan(goal)
            
            # Stream plan steps
            if isinstance(result, dict):
                steps = result.get('steps', [])
                for i, step in enumerate(steps):
                    yield StreamingResponse.stream_sse_event("item", {
                        "content": step,
                        "index": i,
                        "total": len(steps)
                    })
            else:
                yield StreamingResponse.stream_sse_event("chunk", {
                    "content": str(result)
                })
            
            yield StreamingResponse.stream_sse_event("complete", {"status": "ok"})
        
        self._send_stream(plan_generator())
    
    def _handle_explain_with_reasoning_stream(self, body: Dict) -> None:
        """POST /api/explain-with-reasoning?stream - Stream full reasoning trace."""
        query = body.get('query', '')
        if not query:
            self._send_json({'error': 'No query provided'}, 400)
            return
        
        def reasoning_generator():
            yield StreamingResponse.stream_sse_event("start", {
                "operation": "explain_with_reasoning"
            })
            
            explanation = dune.explain_with_reasoning(query)
            reasoning = dune.reason(query)
            
            # Stream explanation
            yield StreamingResponse.stream_sse_event("phase", {
                "name": "explanation",
                "content": str(explanation)[:100]
            })
            
            # Stream reasoning
            words = str(reasoning).split()
            for i, word in enumerate(words[:50]):
                yield StreamingResponse.stream_sse_event("reasoning_chunk", {
                    "content": word,
                    "index": i
                })
            
            yield StreamingResponse.stream_sse_event("complete", {
                "status": "ok",
                "query": query
            })
        
        self._send_stream(reasoning_generator())

    def log_message(self, format, *args) -> None:
        """Override to suppress default logging."""
        print(f"[DUNE Server] {args[0]} {args[1]} {args[2]}")


def run_server(host: str = '0.0.0.0', port: int = 8080) -> None:
    """Run the DUNE web UI server."""
    while True:
        try:
            server = http.server.HTTPServer((host, port), DUNEAPIHandler)
            break
        except OSError as e:
            if e.errno == 98:
                print(f"Port {port} is in use, trying {port + 1}...")
                port += 1
            else:
                raise

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              DUNE-Ω∞-1900-L  Web Interface                   ║
║                                                              ║
║  Learn → Understand → Prove → Simulate → Plan → Explain      ║
║                                                              ║
║  Server running at:                                          ║
║  http://localhost:{port}                                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down DUNE server...")
        server.server_close()


if __name__ == '__main__':
    port = int(os.environ.get('DUNE_PORT', 8080))
    host = os.environ.get('DUNE_HOST', '0.0.0.0')
    run_server(host, port)