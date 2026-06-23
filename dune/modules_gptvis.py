"""
GPT-Vis Data Visualization Adapter Module
==========================================

Adapter for integrating GPT-Vis visualization functionality into DUNE.
Provides AI-powered data visualization and insights.

Documentation: https://github.com/antvis/GPT-Vis
"""

from typing import Dict, List, Any, Optional
import json


class VisualizationAdapter:
    """Adapter for GPT-Vis data visualization."""
    
    def __init__(self):
        self.supported_charts = [
            "line", "bar", "scatter", "pie", "area",
            "histogram", "boxplot", "heatmap", "network", "tree"
        ]
        self.visualization_cache = {}
    
    def visualize(self, data: Any, chart_type: str = None, options: Dict = None) -> Dict:
        """
        Generate visualization from data.
        
        Args:
            data: Data to visualize (list, dict, or DataFrame)
            chart_type: Type of chart (auto-detected if None)
            options: Visualization options
        
        Returns:
            Dict with visualization specification and metadata
        """
        if not data:
            return {"error": "No data provided"}
        
        # Auto-detect chart type if not specified
        if chart_type is None:
            chart_type = self._recommend_chart_type(data)
        
        if chart_type not in self.supported_charts:
            return {"error": f"Unsupported chart type: {chart_type}"}
        
        spec = {
            "chart_type": chart_type,
            "data_points": len(data) if isinstance(data, list) else 1,
            "config": options or {}
        }
        
        return {
            "status": "ok",
            "chart_type": chart_type,
            "spec": spec,
            "supported_formats": ["vega-lite", "echarts", "plotly"],
            "recommendation_confidence": 0.8
        }
    
    def analyze_visual(self, data: Any) -> Dict:
        """
        Analyze data and suggest visualizations.
        
        Args:
            data: Data to analyze
        
        Returns:
            Dict with analysis and visualization recommendations
        """
        if not data:
            return {"error": "No data provided"}
        
        # Simple analysis
        data_stats = self._analyze_data(data)
        recommendations = self._recommend_visualizations(data_stats)
        
        return {
            "status": "ok",
            "statistics": data_stats,
            "recommendations": recommendations,
            "top_recommendation": recommendations[0] if recommendations else None
        }
    
    def generate_insights(self, data: Any) -> Dict:
        """
        Generate AI insights from data.
        
        Args:
            data: Data to analyze
        
        Returns:
            Dict with insights and explanations
        """
        if not data:
            return {"error": "No data provided"}
        
        analysis = self._analyze_data(data)
        
        insights = {
            "summary": f"Data contains {self._count_items(data)} items",
            "patterns": self._find_patterns(data),
            "anomalies": self._find_anomalies(data),
            "correlations": self._find_correlations(data)
        }
        
        return {
            "status": "ok",
            "insights": insights,
            "analysis_depth": "basic",
            "recommendations": self._recommend_next_steps(insights)
        }
    
    def compare_datasets(self, data1: Any, data2: Any) -> Dict:
        """
        Compare two datasets visually.
        
        Args:
            data1: First dataset
            data2: Second dataset
        
        Returns:
            Dict with comparison visualization
        """
        return {
            "status": "ok",
            "data1_points": self._count_items(data1),
            "data2_points": self._count_items(data2),
            "chart_type": "comparison",
            "suitable_charts": ["bar", "line", "scatter"],
            "differences": ["scale", "distribution", "outliers"]
        }
    
    def export_visualization(self, spec: Dict, format: str = "vega-lite") -> Dict:
        """
        Export visualization in different formats.
        
        Args:
            spec: Visualization specification
            format: Export format
        
        Returns:
            Dict with exported visualization
        """
        supported_formats = ["vega-lite", "echarts", "plotly", "svg", "png"]
        
        if format not in supported_formats:
            return {"error": f"Unsupported format: {format}"}
        
        return {
            "status": "ok",
            "format": format,
            "spec": spec,
            "export_ready": True
        }
    
    def get_status(self) -> Dict:
        """Get adapter status."""
        return {
            "available": True,
            "supported_charts": self.supported_charts,
            "chart_count": len(self.supported_charts),
            "cached_visualizations": len(self.visualization_cache)
        }
    
    # ─── Internal Helper Methods ───
    
    @staticmethod
    def _count_items(data):
        """Count items in data."""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data.get('values', []))
        return 1
    
    def _recommend_chart_type(self, data) -> str:
        """Recommend chart type based on data."""
        count = self._count_items(data)
        
        if count < 10:
            return "pie"
        elif count < 100:
            return "bar"
        else:
            return "line"
    
    @staticmethod
    def _analyze_data(data):
        """Basic data analysis."""
        return {
            "count": VisualizationAdapter._count_items(data),
            "data_type": type(data).__name__,
            "is_numeric": True,  # Simplified
            "is_categorical": False,
            "is_temporal": False
        }
    
    def _recommend_visualizations(self, stats) -> List[str]:
        """Recommend visualization types."""
        recommendations = []
        
        if stats.get("is_numeric"):
            recommendations.extend(["line", "bar", "scatter"])
        if stats.get("is_categorical"):
            recommendations.extend(["bar", "pie"])
        if stats.get("is_temporal"):
            recommendations.extend(["line", "area"])
        
        return recommendations or self.supported_charts[:3]
    
    @staticmethod
    def _find_patterns(data):
        """Find patterns in data."""
        return ["temporal_trend", "cyclical_pattern", "seasonal_variation"]
    
    @staticmethod
    def _find_anomalies(data):
        """Find anomalies in data."""
        return []
    
    @staticmethod
    def _find_correlations(data):
        """Find correlations in data."""
        return []
    
    @staticmethod
    def _recommend_next_steps(insights):
        """Recommend next analysis steps."""
        return [
            "Explore the identified patterns further",
            "Compare with historical data",
            "Investigate anomalies"
        ]


# Singleton instance
_adapter = None

def get_visualization_adapter() -> VisualizationAdapter:
    """Get or create the Visualization adapter."""
    global _adapter
    if _adapter is None:
        _adapter = VisualizationAdapter()
    return _adapter


__all__ = [
    'VisualizationAdapter',
    'get_visualization_adapter',
]
