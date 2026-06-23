"""
Haystack RAG Adapter Module
============================

Adapter for integrating Haystack RAG functionality into DUNE.
Provides document retrieval, indexing, and generation capabilities.

Documentation: https://github.com/deepset-ai/haystack
"""

import sys
import os

# Add haystack to path
haystack_path = os.path.join(os.path.dirname(__file__), 'haystack')
if haystack_path not in sys.path:
    sys.path.insert(0, haystack_path)

try:
    # Try importing from the submodule
    from haystack import Pipeline, Document, component
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    class Pipeline:
        pass
    class Document:
        pass


class HaystackRAGAdapter:
    """Adapter for Haystack RAG pipeline."""
    
    def __init__(self):
        self.available = HAYSTACK_AVAILABLE
        if self.available:
            self.document_store = InMemoryDocumentStore()
            self.pipeline = Pipeline()
        else:
            self.document_store = None
            self.pipeline = None
    
    def index_documents(self, documents):
        """Index documents for retrieval."""
        if not self.available:
            return {"error": "Haystack not available"}
        
        docs = [Document(content=doc) if isinstance(doc, str) else doc for doc in documents]
        self.document_store.write_documents(docs)
        return {"status": "ok", "count": len(docs)}
    
    def retrieve(self, query, top_k=5):
        """Retrieve documents matching the query."""
        if not self.available:
            return {"error": "Haystack not available"}
        
        if not self.pipeline and self.document_store:
            retriever = InMemoryBM25Retriever(self.document_store)
            self.pipeline = Pipeline()
            self.pipeline.add_component("retriever", retriever)
        
        if self.pipeline:
            try:
                result = self.pipeline.run({"retriever": {"query": query, "top_k": top_k}})
                return result
            except Exception as e:
                return {"error": str(e)}
        return {}
    
    def generate(self, query, context):
        """Generate answer based on query and context."""
        # This would integrate with an LLM
        return {
            "query": query,
            "context": context,
            "status": "generation not yet configured"
        }


# Singleton instance
_adapter = None

def get_haystack_adapter():
    """Get or create the Haystack adapter."""
    global _adapter
    if _adapter is None:
        _adapter = HaystackRAGAdapter()
    return _adapter


__all__ = [
    'HaystackRAGAdapter',
    'get_haystack_adapter',
    'HAYSTACK_AVAILABLE',
]
