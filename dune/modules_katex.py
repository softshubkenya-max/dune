"""
KaTeX Math Rendering Adapter Module
====================================

Adapter for integrating KaTeX math rendering functionality into DUNE.
Provides LaTeX equation rendering to HTML and SVG.

Documentation: https://github.com/KaTeX/KaTeX
"""

from typing import Dict, Optional
import os
import json


class KaTeXAdapter:
    """Adapter for KaTeX math rendering."""
    
    def __init__(self):
        self.katex_path = os.path.join(os.path.dirname(__file__), 'katex')
        self.options = {
            "throwOnError": False,
            "output": "html"
        }
    
    def render(self, latex_string: str, options: Dict = None) -> Dict:
        """
        Render LaTeX string to HTML.
        
        Args:
            latex_string: LaTeX math expression
            options: KaTeX rendering options
        
        Returns:
            Dict with rendered HTML and metadata
        """
        if not latex_string:
            return {"error": "Empty LaTeX string"}
        
        opts = {**self.options, **(options or {})}
        
        # Mock implementation - in production would call KaTeX CLI or API
        return {
            "status": "ok",
            "input": latex_string,
            "html": f"<span class=\"math\">{latex_string}</span>",
            "options": opts
        }
    
    def render_svg(self, latex_string: str, options: Dict = None) -> Dict:
        """
        Render LaTeX string to SVG.
        
        Args:
            latex_string: LaTeX math expression
            options: KaTeX rendering options
        
        Returns:
            Dict with rendered SVG and metadata
        """
        if not latex_string:
            return {"error": "Empty LaTeX string"}
        
        opts = {**self.options, "output": "mathml", **(options or {})}
        
        return {
            "status": "ok",
            "input": latex_string,
            "svg": f"<svg class=\"math\">{latex_string}</svg>",
            "output_format": "svg",
            "options": opts
        }
    
    def parse_math(self, content: str, delimiter: str = "$") -> Dict:
        """
        Parse mathematical expressions from text content.
        
        Args:
            content: Text content potentially containing math expressions
            delimiter: Math delimiter (default: $)
        
        Returns:
            Dict with parsed expressions
        """
        import re
        
        # Find all math expressions delimited by $ or $$
        single_dollar = re.findall(rf'\${re.escape(delimiter)}?([^\$]+)\${re.escape(delimiter)}?', content)
        expressions = single_dollar
        
        return {
            "status": "ok",
            "content_length": len(content),
            "expressions_found": len(expressions),
            "expressions": expressions[:10]  # Return first 10
        }
    
    def validate_latex(self, latex_string: str) -> Dict:
        """
        Validate LaTeX syntax.
        
        Args:
            latex_string: LaTeX expression to validate
        
        Returns:
            Dict with validation result
        """
        # Simple validation - checks for balanced braces
        open_braces = latex_string.count('{')
        close_braces = latex_string.count('}')
        
        is_valid = open_braces == close_braces
        
        return {
            "status": "ok",
            "valid": is_valid,
            "open_braces": open_braces,
            "close_braces": close_braces,
            "balanced": is_valid
        }
    
    def batch_render(self, expressions: list, format: str = "html") -> Dict:
        """
        Render multiple LaTeX expressions at once.
        
        Args:
            expressions: List of LaTeX expressions
            format: Output format (html or svg)
        
        Returns:
            Dict with rendered expressions
        """
        results = []
        
        for expr in expressions:
            if format == "svg":
                result = self.render_svg(expr)
            else:
                result = self.render(expr)
            results.append(result)
        
        return {
            "status": "ok",
            "count": len(results),
            "format": format,
            "results": results
        }
    
    def get_status(self) -> Dict:
        """Get adapter status."""
        return {
            "available": True,
            "katex_path": self.katex_path,
            "options": self.options
        }


# Singleton instance
_adapter = None

def get_katex_adapter() -> KaTeXAdapter:
    """Get or create the KaTeX adapter."""
    global _adapter
    if _adapter is None:
        _adapter = KaTeXAdapter()
    return _adapter


__all__ = [
    'KaTeXAdapter',
    'get_katex_adapter',
]
