"""
DUNE MCP Autonomous Data Ingestion
====================================
Model Context Protocol autonomous data ingestion.
Crawls and ingests data from MCP servers, APIs, and local sources autonomously.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import re
import threading
import time
import hashlib
from pathlib import Path


@dataclass
class MCPSource:
    """Configuration for an MCP data source."""
    name: str
    source_type: str  # 'mcp', 'api', 'file', 'directory', 'url'
    uri: str
    schedule_seconds: int = 3600  # Default: every hour
    last_fetch: Optional[datetime] = None
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    transform_fn: Optional[Callable] = None

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'source_type': self.source_type,
            'uri': self.uri,
            'schedule_seconds': self.schedule_seconds,
            'last_fetch': self.last_fetch.isoformat() if self.last_fetch else None,
            'enabled': self.enabled,
            'config': self.config,
        }


class MCPDataIngestor:
    """
    Autonomous MCP data ingestion engine.
    Manages multiple data sources and ingests data on schedule.
    """

    def __init__(self):
        self.sources: Dict[str, MCPSource] = {}
        self._ingestion_hooks: Dict[str, List[Callable]] = {
            'on_ingest': [],
            'on_error': [],
            'on_complete': [],
        }
        self._stats: Dict[str, Any] = {
            'total_ingestions': 0,
            'total_errors': 0,
            'total_bytes': 0,
            'last_ingestion': None,
        }

    def register_source(self, source: MCPSource) -> None:
        """Register an MCP data source for autonomous ingestion."""
        self.sources[source.name] = source

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook for ingestion events."""
        if event in self._ingestion_hooks:
            self._ingestion_hooks[event].append(callback)

    def add_file_source(self, name: str, filepath: str,
                        schedule: int = 86400) -> MCPSource:
        """Add a file as a data source."""
        source = MCPSource(
            name=name,
            source_type='file',
            uri=filepath,
            schedule_seconds=schedule,
        )
        self.register_source(source)
        return source

    def add_directory_source(self, name: str, directory: str,
                             pattern: str = "*.txt,*.md,*.py,*.json",
                             schedule: int = 3600) -> MCPSource:
        """Add a directory as a data source."""
        source = MCPSource(
            name=name,
            source_type='directory',
            uri=directory,
            schedule_seconds=schedule,
            config={'pattern': pattern},
        )
        self.register_source(source)
        return source

    def add_url_source(self, name: str, url: str,
                       schedule: int = 7200) -> MCPSource:
        """Add a URL as a data source."""
        source = MCPSource(
            name=name,
            source_type='url',
            uri=url,
            schedule_seconds=schedule,
        )
        self.register_source(source)
        return source

    def add_api_source(self, name: str, api_url: str,
                       headers: Optional[Dict] = None,
                       schedule: int = 300) -> MCPSource:
        """Add an API endpoint as a data source."""
        source = MCPSource(
            name=name,
            source_type='api',
            uri=api_url,
            schedule_seconds=schedule,
            config={'headers': headers or {}},
        )
        self.register_source(source)
        return source

    def add_mcp_source(self, name: str, mcp_server_url: str,
                       mcp_tool: str = "",
                       schedule: int = 3600) -> MCPSource:
        """Add an MCP server as a data source."""
        source = MCPSource(
            name=name,
            source_type='mcp',
            uri=mcp_server_url,
            schedule_seconds=schedule,
            config={'mcp_tool': mcp_tool},
        )
        self.register_source(source)
        return source

    def ingest_source(self, source: MCPSource) -> Optional[str]:
        """
        Ingest data from a single source.
        Returns the ingested text content, or None on failure.
        """
        try:
            content = None

            if source.source_type == 'file':
                content = self._ingest_file(source.uri)
            elif source.source_type == 'directory':
                content = self._ingest_directory(source.uri, source.config.get('pattern', '*'))
            elif source.source_type == 'url':
                content = self._ingest_url(source.uri)
            elif source.source_type == 'api':
                content = self._ingest_api(source.uri, source.config.get('headers', {}))
            elif source.source_type == 'mcp':
                content = self._ingest_mcp(source.uri, source.config.get('mcp_tool', ''))

            if content:
                source.last_fetch = datetime.now()
                self._stats['total_ingestions'] += 1
                self._stats['total_bytes'] += len(content)
                self._stats['last_ingestion'] = datetime.now().isoformat()

                # Apply transform if configured
                if source.transform_fn and callable(source.transform_fn):
                    content = source.transform_fn(content)

                # Fire hooks
                for hook in self._ingestion_hooks['on_ingest']:
                    try:
                        hook(source.name, content)
                    except Exception:
                        pass

                return content

            return None

        except Exception as e:
            self._stats['total_errors'] += 1
            for hook in self._ingestion_hooks['on_error']:
                try:
                    hook(source.name, str(e))
                except Exception:
                    pass
            return None

    def ingest_due_sources(self) -> Dict[str, Optional[str]]:
        """Ingest all sources that are due for refresh."""
        results = {}
        now = datetime.now()

        for name, source in self.sources.items():
            if not source.enabled:
                continue

            # Check if due
            if source.last_fetch:
                elapsed = (now - source.last_fetch).total_seconds()
                if elapsed < source.schedule_seconds:
                    continue

            content = self.ingest_source(source)
            results[name] = content

        # Fire completion hook
        for hook in self._ingestion_hooks['on_complete']:
            try:
                hook(results)
            except Exception:
                pass

        return results

    def ingest_all(self) -> Dict[str, Optional[str]]:
        """Ingest all sources regardless of schedule."""
        results = {}
        for name in self.sources:
            results[name] = self.ingest_source(self.sources[name])
        return results

    def remove_source(self, name: str) -> bool:
        """Remove a data source."""
        if name in self.sources:
            del self.sources[name]
            return True
        return False

    def get_source_status(self) -> List[Dict]:
        """Get status of all sources."""
        statuses = []
        now = datetime.now()

        for name, source in self.sources.items():
            next_fetch = None
            if source.last_fetch:
                next_time = source.last_fetch + timedelta(seconds=source.schedule_seconds)
                next_fetch = next_time.isoformat()

            statuses.append({
                'name': name,
                'type': source.source_type,
                'uri': source.uri,
                'enabled': source.enabled,
                'last_fetch': source.last_fetch.isoformat() if source.last_fetch else None,
                'next_fetch': next_fetch,
                'schedule_seconds': source.schedule_seconds,
            })

        return statuses

    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return dict(self._stats)

    def _ingest_file(self, filepath: str) -> Optional[str]:
        """Ingest a file."""
        path = Path(filepath)
        if not path.exists():
            return None

        try:
            return path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            # Try binary read
            try:
                data = path.read_bytes()
                return data.decode('utf-8', errors='replace')
            except Exception:
                return None

    def _ingest_directory(self, directory: str, pattern: str) -> Optional[str]:
        """Ingest all files in a directory."""
        patterns = [p.strip() for p in pattern.split(',')]
        path = Path(directory)
        all_content = []

        for pat in patterns:
            for f in path.rglob(pat):
                if f.is_file():
                    try:
                        content = f.read_text(encoding='utf-8', errors='replace')
                        all_content.append(f"--- File: {f} ---\n{content}")
                    except Exception:
                        pass

        return '\n\n'.join(all_content) if all_content else None

    def _ingest_url(self, url: str) -> Optional[str]:
        """Ingest a URL."""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=15) as resp:
                content = resp.read().decode('utf-8', errors='replace')
                # Strip HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content[:500000]  # Max 500k chars
        except Exception:
            return None

    def _ingest_api(self, api_url: str, headers: Dict) -> Optional[str]:
        """Ingest from an API endpoint."""
        try:
            import urllib.request
            req = urllib.request.Request(api_url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode('utf-8', errors='replace')
                # Try to pretty-print JSON
                try:
                    parsed = json.loads(data)
                    return json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    return data[:500000]
        except Exception:
            return None

    def _ingest_mcp(self, mcp_url: str, tool: str) -> Optional[str]:
        """Ingest from an MCP server endpoint."""
        try:
            import urllib.request
            # Standard MCP endpoint for tools/list
            tools_url = mcp_url.rstrip('/') + '/tools/list'
            req = urllib.request.Request(tools_url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode('utf-8', errors='replace')
                return f"MCP Server: {mcp_url}\n\nTools:\n{data[:500000]}"
        except Exception:
            # Try just fetching the base URL
            try:
                with urllib.request.urlopen(mcp_url, timeout=15) as resp:
                    return resp.read().decode('utf-8', errors='replace')[:500000]
            except Exception:
                return None

    def to_dict(self) -> Dict:
        return {
            'stats': self.get_stats(),
            'sources': {k: v.to_dict() for k, v in self.sources.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
# II. AUTONOMOUS SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class AutonomousScheduler:
    """
    Autonomous scheduler that continuously ingests data on a background thread.
    Runs data collection cycles, learns from ingested data, and adapts.
    """

    def __init__(self, dune_instance=None):
        self.ingestor = MCPDataIngestor()
        self._dune = dune_instance
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cycle_count = 0
        self._auto_learn = True
        self._cycle_interval = 60  # Check every 60 seconds
        self._last_learn_time: Dict[str, datetime] = {}

    def attach_dune(self, dune_instance) -> None:
        """Attach a DUNE instance for automatic learning."""
        self._dune = dune_instance

    def start(self, daemon: bool = True) -> None:
        """Start the autonomous scheduler in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=daemon)
        self._thread.start()

    def stop(self) -> None:
        """Stop the autonomous scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def run_once(self) -> Dict[str, Any]:
        """Run a single ingestion cycle."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'ingested': {},
            'learned': 0,
            'errors': 0,
        }

        # Ingest due sources
        ingested = self.ingestor.ingest_due_sources()
        results['ingested'] = {
            name: len(content) if content else 0
            for name, content in ingested.items()
        }

        # Auto-learn from ingested content
        if self._auto_learn and self._dune:
            for name, content in ingested.items():
                if content and len(content) > 20:
                    try:
                        # Feed to DUNE-1900
                        learn_result = self._dune.learn(
                            content[:50000],  # Limit per chunk
                            source=f"autonomous/{name}"
                        )
                        results['learned'] += learn_result['facts_stored']
                        self._last_learn_time[name] = datetime.now()
                    except Exception:
                        results['errors'] += 1

        self._cycle_count += 1
        return results

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self.run_once()
            except Exception:
                pass
            # Wait for next cycle
            for _ in range(self._cycle_interval):
                if not self._running:
                    break
                time.sleep(1)

    def configure_autonomous_mode(self, mode: str = 'balanced') -> None:
        """
        Configure the autonomous mode:
        - 'aggressive': Check every 30s, ingest all sources regardless
        - 'balanced': Check every 60s, only due sources
        - 'conservative': Check every 300s, only due sources
        """
        mode = mode.lower()
        if mode == 'aggressive':
            self._cycle_interval = 30
        elif mode == 'balanced':
            self._cycle_interval = 60
        elif mode == 'conservative':
            self._cycle_interval = 300
        else:
            self._cycle_interval = 60

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'running': self._running,
            'cycle_count': self._cycle_count,
            'cycle_interval': self._cycle_interval,
            'auto_learn': self._auto_learn,
            'sources': self.ingestor.get_source_status(),
            'ingestion_stats': self.ingestor.get_stats(),
        }

    def to_dict(self) -> Dict:
        return {
            'running': self._running,
            'cycle_count': self._cycle_count,
            'cycle_interval': self._cycle_interval,
            'auto_learn': self._auto_learn,
            'ingestor': self.ingestor.to_dict(),
        }