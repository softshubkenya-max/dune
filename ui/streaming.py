"""
Streaming Response Helper
=========================

Utilities for streaming responses from DUNE operations.
Supports Server-Sent Events (SSE) and chunked transfer encoding.
"""

import json
from typing import Any, Dict, Callable, Iterator, Optional
import threading
import time


class StreamingResponse:
    """Helper for streaming responses to clients."""
    
    @staticmethod
    def stream_sse_event(event_type: str, data: Any) -> str:
        """
        Format data as Server-Sent Event.
        
        Args:
            event_type: Type of event (e.g., 'chunk', 'complete', 'error')
            data: Data to send
        
        Returns:
            SSE formatted string
        """
        json_data = json.dumps(data, default=str)
        return f"event: {event_type}\ndata: {json_data}\n\n"
    
    @staticmethod
    def stream_generator(operation: Callable, *args, **kwargs) -> Iterator[str]:
        """
        Create a generator that streams operation results.
        
        Args:
            operation: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Yields:
            SSE formatted events
        """
        try:
            # Send start event
            yield StreamingResponse.stream_sse_event("start", {
                "status": "starting",
                "timestamp": time.time()
            })
            
            # Execute operation
            result = operation(*args, **kwargs)
            
            # Handle different result types
            if isinstance(result, str):
                # Stream string character by character or word by word
                words = result.split()
                for i, word in enumerate(words):
                    yield StreamingResponse.stream_sse_event("chunk", {
                        "content": word,
                        "index": i,
                        "total": len(words)
                    })
                    time.sleep(0.01)  # Small delay for streaming effect
            
            elif isinstance(result, dict):
                # Stream dict key by key
                for key, value in result.items():
                    yield StreamingResponse.stream_sse_event("data", {
                        "key": key,
                        "value": value
                    })
            
            elif isinstance(result, (list, tuple)):
                # Stream list items
                for i, item in enumerate(result):
                    yield StreamingResponse.stream_sse_event("item", {
                        "content": item,
                        "index": i,
                        "total": len(result)
                    })
            
            else:
                # Send result as single chunk
                yield StreamingResponse.stream_sse_event("chunk", {
                    "content": str(result)
                })
            
            # Send complete event
            yield StreamingResponse.stream_sse_event("complete", {
                "status": "complete",
                "timestamp": time.time()
            })
        
        except Exception as e:
            yield StreamingResponse.stream_sse_event("error", {
                "error": str(e),
                "type": type(e).__name__
            })


class IncrementalResponse:
    """Helper for incremental response building."""
    
    def __init__(self):
        self.chunks = []
        self.metadata = {}
    
    def add_chunk(self, content: Any, metadata: Dict = None):
        """Add a chunk of content."""
        self.chunks.append(content)
        if metadata:
            self.metadata.update(metadata)
    
    def stream(self) -> Iterator[str]:
        """Stream accumulated chunks."""
        for i, chunk in enumerate(self.chunks):
            event_data = {
                "content": chunk,
                "index": i,
                "total": len(self.chunks)
            }
            yield StreamingResponse.stream_sse_event("chunk", event_data)
    
    def get_complete(self) -> Dict:
        """Get complete accumulated response."""
        return {
            "content": "".join(str(c) for c in self.chunks),
            "chunks": len(self.chunks),
            "metadata": self.metadata
        }


class ProgressTracker:
    """Track and report progress of long operations."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.callbacks = []
    
    def on_progress(self, callback: Callable):
        """Register a callback for progress updates."""
        self.callbacks.append(callback)
    
    def step(self, message: str = None):
        """Mark a step as complete."""
        self.current_step += 1
        progress_data = {
            "step": self.current_step,
            "total": self.total_steps,
            "percentage": (self.current_step / self.total_steps) * 100,
            "message": message
        }
        
        for callback in self.callbacks:
            callback(progress_data)
        
        return progress_data
    
    def stream_progress(self) -> Iterator[str]:
        """Generate progress events."""
        def emit_progress(data):
            yield StreamingResponse.stream_sse_event("progress", data)
        
        self.on_progress(lambda data: list(emit_progress(data)))


def stream_with_progress(operation: Callable, total_steps: int) -> Iterator[str]:
    """
    Stream an operation with progress tracking.
    
    Args:
        operation: Function to execute (should accept progress tracker)
        total_steps: Expected number of steps
    
    Yields:
        SSE formatted events with progress
    """
    tracker = ProgressTracker(total_steps)
    
    try:
        yield StreamingResponse.stream_sse_event("start", {
            "total_steps": total_steps
        })
        
        # Execute operation with progress tracking
        result = operation(tracker)
        
        yield StreamingResponse.stream_sse_event("complete", {
            "result": result,
            "total_steps": total_steps
        })
    
    except Exception as e:
        yield StreamingResponse.stream_sse_event("error", {
            "error": str(e),
            "step": tracker.current_step
        })


__all__ = [
    'StreamingResponse',
    'IncrementalResponse',
    'ProgressTracker',
    'stream_with_progress',
]
