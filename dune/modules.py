"""
DUNE Modules Loader
===================

Unified interface for loading and managing all integrated modules.
Provides centralized access to all adapter modules and their capabilities.
"""

from typing import Dict, Any, List

# Import all module adapters
from dune.modules_haystack import get_haystack_adapter, HAYSTACK_AVAILABLE
from dune.modules_semantic_kernel import get_semantic_kernel_adapter, SEMANTIC_KERNEL_AVAILABLE
from dune.modules_botpress import get_botpress_adapter
from dune.modules_katex import get_katex_adapter
from dune.modules_markdown import get_markdown_adapter
from dune.modules_gptvis import get_visualization_adapter
from dune.modules_generative_ui import get_generative_ui_adapter


class ModulesManager:
    """Central manager for all DUNE modules."""
    
    def __init__(self):
        self.modules = {}
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all available modules."""
        self.modules = {
            'haystack': {
                'name': 'Haystack RAG',
                'adapter': get_haystack_adapter,
                'available': HAYSTACK_AVAILABLE,
                'category': 'retrieval',
                'description': 'Retrieval-Augmented Generation pipeline'
            },
            'semantic_kernel': {
                'name': 'Semantic Kernel',
                'adapter': get_semantic_kernel_adapter,
                'available': SEMANTIC_KERNEL_AVAILABLE,
                'category': 'orchestration',
                'description': 'AI orchestration and plugin management'
            },
            'botpress': {
                'name': 'Botpress',
                'adapter': get_botpress_adapter,
                'available': True,
                'category': 'conversational_ai',
                'description': 'Conversational AI platform'
            },
            'katex': {
                'name': 'KaTeX',
                'adapter': get_katex_adapter,
                'available': True,
                'category': 'math',
                'description': 'Mathematical notation rendering'
            },
            'markdown': {
                'name': 'LLM-Markdown',
                'adapter': get_markdown_adapter,
                'available': True,
                'category': 'text_processing',
                'description': 'Markdown processing with LLM integration'
            },
            'visualization': {
                'name': 'GPT-Vis',
                'adapter': get_visualization_adapter,
                'available': True,
                'category': 'visualization',
                'description': 'AI-powered data visualization'
            },
            'generative_ui': {
                'name': 'Generative UI',
                'adapter': get_generative_ui_adapter,
                'available': True,
                'category': 'ui',
                'description': 'UI components for generative AI apps'
            }
        }
    
    def get_module(self, module_name: str) -> Any:
        """
        Get a specific module adapter.
        
        Args:
            module_name: Name of the module
        
        Returns:
            Module adapter instance
        """
        if module_name not in self.modules:
            raise ValueError(f"Module not found: {module_name}")
        
        module_info = self.modules[module_name]
        if not module_info['available']:
            raise RuntimeError(f"Module not available: {module_name}")
        
        return module_info['adapter']()
    
    def list_modules(self) -> Dict:
        """List all available modules."""
        return {
            "total": len(self.modules),
            "modules": {
                name: {
                    'name': info['name'],
                    'available': info['available'],
                    'category': info['category'],
                    'description': info['description']
                }
                for name, info in self.modules.items()
            }
        }
    
    def list_by_category(self, category: str) -> Dict:
        """List modules by category."""
        filtered = {
            name: info for name, info in self.modules.items()
            if info['category'] == category
        }
        
        return {
            "category": category,
            "modules": list(filtered.keys()),
            "count": len(filtered)
        }
    
    def get_status(self) -> Dict:
        """Get status of all modules."""
        status = {}
        
        for name, info in self.modules.items():
            if info['available']:
                try:
                    adapter = info['adapter']()
                    module_status = adapter.get_status()
                except Exception as e:
                    module_status = {"error": str(e)}
            else:
                module_status = {"available": False}
            
            status[name] = module_status
        
        return {
            "timestamp": self._get_timestamp(),
            "modules": status,
            "available_count": sum(1 for info in self.modules.values() if info['available'])
        }
    
    def get_capabilities(self) -> Dict:
        """Get all capabilities across all modules."""
        capabilities = {}
        
        for name, info in self.modules.items():
            capabilities[name] = {
                'name': info['name'],
                'category': info['category'],
                'description': info['description'],
                'available': info['available']
            }
        
        return capabilities
    
    @staticmethod
    def _get_timestamp():
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Global instance
_manager = None

def get_modules_manager() -> ModulesManager:
    """Get or create the global modules manager."""
    global _manager
    if _manager is None:
        _manager = ModulesManager()
    return _manager


def get_module(module_name: str) -> Any:
    """
    Convenience function to get a module.
    
    Args:
        module_name: Name of the module
    
    Returns:
        Module adapter instance
    """
    manager = get_modules_manager()
    return manager.get_module(module_name)


def list_modules() -> Dict:
    """List all available modules."""
    manager = get_modules_manager()
    return manager.list_modules()


def get_module_status() -> Dict:
    """Get status of all modules."""
    manager = get_modules_manager()
    return manager.get_status()


__all__ = [
    'ModulesManager',
    'get_modules_manager',
    'get_module',
    'list_modules',
    'get_module_status',
]
