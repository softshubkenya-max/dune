"""
Generative UI Components Adapter Module
========================================

Adapter for integrating Awesome Generative UI components into DUNE.
Provides UI patterns and components for generative AI applications.

Documentation: https://github.com/narrowin/awesome-generative-ui
"""

from typing import Dict, List, Any, Optional


class GenerativeUIAdapter:
    """Adapter for Generative UI components and patterns."""
    
    COMPONENTS = {
        "chat_interface": {
            "name": "Chat Interface",
            "description": "Real-time chat UI with message streaming",
            "features": ["typing_indicator", "markdown_support", "code_highlight", "emoji_support"]
        },
        "prompt_input": {
            "name": "Prompt Input",
            "description": "Advanced input component for LLM prompts",
            "features": ["syntax_highlighting", "autocomplete", "templates", "history"]
        },
        "response_display": {
            "name": "Response Display",
            "description": "Display LLM responses with formatting",
            "features": ["streaming", "citations", "code_blocks", "interactive"]
        },
        "token_counter": {
            "name": "Token Counter",
            "description": "Display token usage and cost estimation",
            "features": ["live_counting", "cost_estimation", "model_selection"]
        },
        "conversation_history": {
            "name": "Conversation History",
            "description": "Manage and display conversation threads",
            "features": ["search", "export", "share", "branch_conversations"]
        },
        "sidebar": {
            "name": "Sidebar",
            "description": "Navigation and context sidebar",
            "features": ["collapsible", "themes", "responsive", "keyboard_nav"]
        },
        "settings_panel": {
            "name": "Settings Panel",
            "description": "Configure LLM parameters and UI preferences",
            "features": ["presets", "sliders", "input_validation", "save_preferences"]
        }
    }
    
    PATTERNS = {
        "streaming_response": {
            "name": "Streaming Response",
            "description": "Stream LLM responses in real-time",
            "implementation": "event_based",
            "techniques": ["chunked_transfer", "websockets", "sse"]
        },
        "multi_turn_conversation": {
            "name": "Multi-turn Conversation",
            "description": "Manage context across multiple turns",
            "implementation": "state_machine",
            "techniques": ["context_window", "summarization", "relevance_scoring"]
        },
        "parallel_requests": {
            "name": "Parallel Requests",
            "description": "Handle multiple concurrent LLM requests",
            "implementation": "async",
            "techniques": ["task_queuing", "rate_limiting", "batching"]
        },
        "error_recovery": {
            "name": "Error Recovery",
            "description": "Gracefully handle and recover from errors",
            "implementation": "retry_strategy",
            "techniques": ["exponential_backoff", "fallback_models", "user_notification"]
        },
        "data_visualization": {
            "name": "Data Visualization",
            "description": "Visualize data returned by LLMs",
            "implementation": "chart_library",
            "techniques": ["auto_detection", "interactive_charts", "export"]
        },
        "model_comparison": {
            "name": "Model Comparison",
            "description": "Compare outputs from different models",
            "implementation": "side_by_side",
            "techniques": ["split_view", "diff_highlighting", "metrics"]
        }
    }
    
    def __init__(self):
        self.available = True
        self.custom_components = {}
        self.theme = "light"
    
    def get_component(self, component_type: str) -> Dict:
        """
        Get a UI component specification.
        
        Args:
            component_type: Type of component
        
        Returns:
            Dict with component specification
        """
        if component_type in self.COMPONENTS:
            return {
                "status": "ok",
                "type": component_type,
                "component": self.COMPONENTS[component_type]
            }
        elif component_type in self.custom_components:
            return {
                "status": "ok",
                "type": component_type,
                "component": self.custom_components[component_type],
                "custom": True
            }
        else:
            return {"error": f"Component not found: {component_type}"}
    
    def get_pattern(self, pattern_name: str) -> Dict:
        """
        Get a design pattern specification.
        
        Args:
            pattern_name: Name of the pattern
        
        Returns:
            Dict with pattern specification
        """
        if pattern_name in self.PATTERNS:
            return {
                "status": "ok",
                "pattern": pattern_name,
                "specification": self.PATTERNS[pattern_name]
            }
        else:
            return {"error": f"Pattern not found: {pattern_name}"}
    
    def list_components(self) -> Dict:
        """List all available components."""
        return {
            "status": "ok",
            "components": list(self.COMPONENTS.keys()),
            "custom_components": list(self.custom_components.keys()),
            "total": len(self.COMPONENTS) + len(self.custom_components)
        }
    
    def list_patterns(self) -> Dict:
        """List all available patterns."""
        return {
            "status": "ok",
            "patterns": list(self.PATTERNS.keys()),
            "total": len(self.PATTERNS)
        }
    
    def register_custom_component(self, name: str, spec: Dict) -> Dict:
        """
        Register a custom component.
        
        Args:
            name: Component name
            spec: Component specification
        
        Returns:
            Dict with registration status
        """
        self.custom_components[name] = spec
        return {
            "status": "ok",
            "component": name,
            "registered": True
        }
    
    def set_theme(self, theme: str) -> Dict:
        """
        Set the UI theme.
        
        Args:
            theme: Theme name (light, dark, auto)
        
        Returns:
            Dict with theme status
        """
        valid_themes = ["light", "dark", "auto", "custom"]
        
        if theme not in valid_themes:
            return {"error": f"Invalid theme: {theme}"}
        
        self.theme = theme
        return {
            "status": "ok",
            "theme": theme,
            "applied": True
        }
    
    def get_responsive_config(self, breakpoint: str = "desktop") -> Dict:
        """
        Get responsive design configuration.
        
        Args:
            breakpoint: Screen size breakpoint
        
        Returns:
            Dict with responsive configuration
        """
        breakpoints = {
            "mobile": {"width": "100%", "max_width": "480px"},
            "tablet": {"width": "100%", "max_width": "768px"},
            "desktop": {"width": "100%", "max_width": "1920px"}
        }
        
        return {
            "status": "ok",
            "breakpoint": breakpoint,
            "config": breakpoints.get(breakpoint, breakpoints["desktop"]),
            "all_breakpoints": list(breakpoints.keys())
        }
    
    def get_accessibility_config(self) -> Dict:
        """Get accessibility configuration."""
        return {
            "status": "ok",
            "features": {
                "keyboard_navigation": True,
                "screen_reader_support": True,
                "high_contrast_mode": True,
                "reduced_motion": True,
                "text_scaling": True,
                "focus_indicators": True
            }
        }
    
    def export_component_library(self) -> Dict:
        """Export all components as a library."""
        return {
            "status": "ok",
            "components": self.COMPONENTS,
            "patterns": self.PATTERNS,
            "custom_components": self.custom_components,
            "total_components": len(self.COMPONENTS) + len(self.custom_components),
            "format": "openapi_3.0"
        }
    
    def get_status(self) -> Dict:
        """Get adapter status."""
        return {
            "available": self.available,
            "built_in_components": len(self.COMPONENTS),
            "custom_components": len(self.custom_components),
            "patterns": len(self.PATTERNS),
            "theme": self.theme
        }


# Singleton instance
_adapter = None

def get_generative_ui_adapter() -> GenerativeUIAdapter:
    """Get or create the Generative UI adapter."""
    global _adapter
    if _adapter is None:
        _adapter = GenerativeUIAdapter()
    return _adapter


__all__ = [
    'GenerativeUIAdapter',
    'get_generative_ui_adapter',
]
