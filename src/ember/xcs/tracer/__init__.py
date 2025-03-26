# Export the autograph context manager for public API
from contextlib import contextmanager

from ._context_types import TraceContextData
from .autograph import AutoGraphBuilder
from .xcs_tracing import TracerContext, TraceRecord
from .unified_jit import jit


@contextmanager
def autograph(*args, **kwargs):
    """Context manager for automatic graph building.

    Creates a computation graph context where operations are automatically
    recorded as graph nodes rather than being executed immediately.

    Yields:
        An XCSGraph object that can be used to execute the recorded operations
    """
    from ember.xcs.graph.xcs_graph import XCSGraph

    graph = XCSGraph()
    try:
        yield graph
    finally:
        # When exiting context, the graph has been built and can be used
        pass


# Expose individual JIT implementations for backwards compatibility
from .tracer_decorator import jit as trace_jit
from .structural_jit import structural_jit, disable_structural_jit


__all__ = [
    "TracerContext",
    "TraceRecord",
    "TraceContextData",
    "AutoGraphBuilder",
    "autograph",
    "jit",  # Unified JIT interface
    "trace_jit",  # Legacy access to trace-based JIT
    "structural_jit",  # Legacy access to structural JIT
    "disable_structural_jit",  # Utility for disabling structural JIT
]
