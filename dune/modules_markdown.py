"""
LLM-Markdown Adapter Module
===========================

Adapter for integrating LLM-Markdown functionality into DUNE.
Provides Markdown processing with embedded LLM capabilities.

Documentation: https://github.com/skovy/llm-markdown
"""

from typing import Dict, List, Any, Optional
import re


class MarkdownAdapter:
    """Adapter for LLM-Markdown processing."""
    
    def __init__(self):
        self.llm_call_pattern = r'\{\{(.*?)\}\}'
        self.code_fence_pattern = r'```(\w+)?\n(.*?)\n```'
        self.template_vars = {}
    
    def process(self, markdown_content: str) -> Dict:
        """
        Process markdown with LLM capabilities.
        
        Args:
            markdown_content: Markdown text with potential LLM calls
        
        Returns:
            Dict with processed content and metadata
        """
        # Find embedded LLM calls
        llm_calls = re.findall(self.llm_call_pattern, markdown_content)
        
        # Find code blocks
        code_blocks = re.findall(self.code_fence_pattern, markdown_content, re.DOTALL)
        
        return {
            "status": "ok",
            "original_length": len(markdown_content),
            "llm_calls_found": len(llm_calls),
            "code_blocks_found": len(code_blocks),
            "llm_calls": llm_calls[:5],  # First 5
            "processed_content": markdown_content
        }
    
    def embed_llm_calls(self, content: str, llm_queries: Dict[str, str]) -> Dict:
        """
        Embed LLM calls into markdown.
        
        Args:
            content: Markdown content
            llm_queries: Dict of {placeholder: query}
        
        Returns:
            Dict with modified content and embeddings
        """
        modified = content
        embeddings = {}
        
        for placeholder, query in llm_queries.items():
            # Replace placeholder with LLM call syntax
            llm_call = f"{{{{ {query} }}}}"
            modified = modified.replace(f"{{{placeholder}}}", llm_call)
            embeddings[placeholder] = query
        
        return {
            "status": "ok",
            "modified_content": modified,
            "embeddings": embeddings,
            "count": len(embeddings)
        }
    
    def generate_from_template(self, template: str, context: Dict) -> Dict:
        """
        Generate content from markdown template with context.
        
        Args:
            template: Markdown template with variables
            context: Dict of variables to substitute
        
        Returns:
            Dict with generated content
        """
        generated = template
        
        # Replace variables in template
        for key, value in context.items():
            generated = generated.replace(f"{{{{{key}}}}}", str(value))
        
        return {
            "status": "ok",
            "template_length": len(template),
            "generated_length": len(generated),
            "variables_substituted": len(context),
            "generated": generated
        }
    
    def parse_frontmatter(self, markdown_content: str) -> Dict:
        """
        Parse YAML frontmatter from markdown.
        
        Args:
            markdown_content: Markdown with potential YAML frontmatter
        
        Returns:
            Dict with frontmatter and content
        """
        if markdown_content.startswith('---'):
            parts = markdown_content.split('---', 2)
            if len(parts) >= 3:
                frontmatter_raw = parts[1].strip()
                content = parts[2].strip()
                
                # Parse simple YAML (not a full parser)
                frontmatter = {}
                for line in frontmatter_raw.split('\n'):
                    if ':' in line:
                        key, val = line.split(':', 1)
                        frontmatter[key.strip()] = val.strip()
                
                return {
                    "status": "ok",
                    "has_frontmatter": True,
                    "frontmatter": frontmatter,
                    "content": content
                }
        
        return {
            "status": "ok",
            "has_frontmatter": False,
            "frontmatter": {},
            "content": markdown_content
        }
    
    def extract_headings(self, markdown_content: str) -> Dict:
        """
        Extract heading hierarchy from markdown.
        
        Args:
            markdown_content: Markdown content
        
        Returns:
            Dict with heading structure
        """
        headings = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', markdown_content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2)
            headings.append({
                "level": level,
                "text": text
            })
        
        return {
            "status": "ok",
            "heading_count": len(headings),
            "headings": headings
        }
    
    def set_context(self, key: str, value: Any):
        """Set template context variable."""
        self.template_vars[key] = value
        return {
            "status": "ok",
            "key": key,
            "value": value
        }
    
    def get_status(self) -> Dict:
        """Get adapter status."""
        return {
            "available": True,
            "llm_call_pattern": self.llm_call_pattern,
            "template_vars": len(self.template_vars)
        }


# Singleton instance
_adapter = None

def get_markdown_adapter() -> MarkdownAdapter:
    """Get or create the Markdown adapter."""
    global _adapter
    if _adapter is None:
        _adapter = MarkdownAdapter()
    return _adapter


__all__ = [
    'MarkdownAdapter',
    'get_markdown_adapter',
]
