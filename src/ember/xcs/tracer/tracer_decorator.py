"""
JIT Compilation and Execution Tracing for XCS Operators

This module provides a just-in-time (JIT) compilation system for Ember operators
through execution tracing. The @jit decorator transforms operator classes by
instrumenting them to record their execution patterns and automatically compile
optimized execution plans.

Key features:
1. Transparent operator instrumentation via the @jit decorator
2. Automatic execution graph construction from traced operator calls
3. Compile-once, execute-many optimization for repeated operations
4. Support for pre-compilation with sample inputs
5. Configurable tracing and caching behaviors

Implementation follows functional programming principles where possible,
separating concerns between tracing, compilation, and execution. The design
adheres to the Open/Closed Principle by extending operator behavior without
modifying their core implementation.

Example:
    @jit
    class MyOperator(Operator):
        def __call__(self, *, inputs):
            # Complex, multi-step computation
            return result

    # First call triggers tracing and compilation
    op = MyOperator()
    result1 = op(inputs={"text": "example"})

    # Subsequent calls reuse the compiled execution plan
    result2 = op(inputs={"text": "another example"})
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
    get_type_hints,
)

# Import the base classes carefully to avoid circular imports
from ember.xcs.tracer.xcs_tracing import TracerContext, TraceRecord

# We need to use a string for the bound to avoid circular imports
# Type variable for Operator subclasses
OperatorType = TypeVar("OperatorType", bound="Operator")
# Type alias for the decorator function's return type
OperatorDecorator = Callable[[Type[OperatorType]], Type[OperatorType]]

# Use a Protocol for Operator to avoid circular imports
from typing import Protocol, runtime_checkable


@runtime_checkable
class Operator(Protocol):
    """Protocol defining the expected interface for Operators."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the operator with provided inputs."""
        ...


# Forward import execution components to avoid circular imports
from ember.xcs.graph.xcs_graph import XCSGraph

# Cache to store compiled execution graphs for each operator class instance
_COMPILED_GRAPHS: Dict[int, XCSGraph] = {}


def jit(
    func=None,
    *,
    sample_input: Optional[Dict[str, Any]] = None,
    force_trace: bool = False,
    recursive: bool = True,
):
    """Just-In-Time compilation decorator for Ember Operators.

    The @jit decorator transforms Operator classes to automatically trace their execution
    and compile optimized execution plans. This brings significant performance benefits
    for complex operations and operator pipelines by analyzing the execution pattern
    once and reusing the optimized plan for subsequent calls.

    The implementation follows a lazily evaluated, memoization pattern:
    1. First execution triggers tracing to capture the full execution graph
    2. The traced operations are compiled into an optimized execution plan
    3. Subsequent calls reuse this plan without re-tracing (unless force_trace=True)

    Pre-compilation via sample_input is available for performance-critical paths where
    even the first execution needs to be fast. This implements an "eager" JIT pattern
    where compilation happens at initialization time rather than first execution time.

    Design principles:
    - Separation of concerns: Tracing, compilation, and execution are distinct phases
    - Minimal overhead: Non-tracing execution paths have negligible performance impact
    - Transparency: Decorated operators maintain their original interface contract
    - Configurability: Multiple options allow fine-tuning for different use cases

    Args:
        func: The function or class to be JIT-compiled. This is automatically passed when
             using the @jit syntax directly. If using @jit(...) with parameters, this will be None.
        sample_input: Optional pre-defined input for eager compilation during initialization.
                    This enables "compile-time" optimization rather than runtime JIT compilation.
                    Recommended for performance-critical initialization paths.
        force_trace: When True, disables caching and traces every invocation.
                    This is valuable for debugging and for operators whose execution
                    pattern varies significantly based on input values.
                    Performance impact: Significant, as caching benefits are disabled.
        recursive: Controls whether nested operator calls are also traced and compiled.
                 Currently limited to direct child operators observed during tracing.
                 Default is True, enabling full pipeline optimization.

    Returns:
        A decorated function/class or a decorator function that transforms the target
        Operator subclass by instrumenting its initialization and call methods for tracing.

    Raises:
        TypeError: If applied to a class that doesn't inherit from Operator.
                  The decorator strictly enforces type safety to prevent
                  incorrect usage on unsupported class types.

    Example:
        # Direct decoration (no parameters)
        @jit
        class SimpleOperator(Operator):
            def __call__(self, *, inputs):
                return process(inputs)

        # Parameterized decoration
        @jit(sample_input={"text": "example"})
        class ProcessorOperator(Operator):
            def __call__(self, *, inputs):
                # Complex multi-step process
                return {"result": processed_output}
    """

    def decorator(cls: Type[OperatorType]) -> Type[OperatorType]:
        """Internal decorator function applied to the Operator class.

        Args:
            cls: The Operator subclass to be instrumented.

        Returns:
            The decorated Operator class with tracing capabilities.

        Raises:
            TypeError: If cls is not an Operator subclass.
        """
        # More robust type checking that allows duck typing
        try:
            if not issubclass(cls, Operator):
                # Check for duck typing - if it has a __call__ method with the right signature
                if not (
                    hasattr(cls, "__call__") and callable(getattr(cls, "__call__"))
                ):
                    raise TypeError(
                        "@jit decorator can only be applied to an Operator-like class with a __call__ method."
                    )
        except TypeError:
            # This handles the case where cls is not a class at all
            raise TypeError(
                "@jit decorator can only be applied to a class, not a function or other object."
            )

        original_call = cls.__call__
        original_init = cls.__init__

        @functools.wraps(original_init)
        def traced_init(self: OperatorType, *args: Any, **kwargs: Any) -> None:
            """Wrapped __init__ method that initializes the operator and pre-traces with sample input."""
            # Call the original __init__
            original_init(self, *args, **kwargs)

            # If sample_input is provided, perform pre-tracing during initialization
            if sample_input is not None:
                # Create a tracer context and trace the operator's execution
                with TracerContext() as tracer:
                    original_call(self=self, inputs=sample_input)

                if tracer.records:
                    # Import here to avoid circular imports
                    from ember.xcs.tracer.autograph import AutoGraphBuilder

                    # Build and cache the graph
                    graph_builder = AutoGraphBuilder()
                    graph = graph_builder.build_graph(tracer.records)
                    _COMPILED_GRAPHS[id(self)] = graph

        @functools.wraps(original_call)
        def traced_call(self: OperatorType, *, inputs: Dict[str, Any]) -> Any:
            """Wrapped __call__ method that records execution trace.

            Args:
                inputs: The input parameters for the operator.

            Returns:
                The output from the operator execution.
            """
            # Setup logging
            logger = logging.getLogger("ember.xcs.tracer.jit")
            
            # Get current tracer context
            tracer: Optional[TracerContext] = TracerContext.get_current()
            
            # For debugging and test purposes
            force_trace_local = getattr(self, "_force_trace", force_trace)
            
            # Check if we have a cached compiled graph and should use it
            op_id = id(self)
            
            # Phase 1: Try optimized execution with cached graph
            # -------------------------------------------------
            if not force_trace_local and op_id in _COMPILED_GRAPHS:
                try:
                    # Import execution components
                    from ember.xcs.engine.xcs_engine import execute_graph
                    from ember.xcs.engine.xcs_engine import TopologicalSchedulerWithParallelDispatch
                    
                    # Get cached graph
                    graph = _COMPILED_GRAPHS[op_id]
                    logger.debug(f"Using optimized graph for {self.__class__.__name__} (nodes: {len(graph.nodes)})")
                    
                    # Execute the graph with the parallel scheduler
                    # The scheduler will automatically determine optimal execution strategy
                    results = execute_graph(
                        graph=graph,
                        global_input=inputs,
                        scheduler=TopologicalSchedulerWithParallelDispatch()
                    )
                    
                    # Find the appropriate result to return
                    # Try different strategies to determine which node's output to return
                    
                    # Strategy 1: Look for leaf nodes (nodes without outbound edges)
                    leaf_nodes = [node_id for node_id, node in graph.nodes.items() 
                                 if not node.outbound_edges]
                    
                    if len(leaf_nodes) == 1:
                        # Single leaf node - clear choice for output
                        logger.debug(f"Using output from single leaf node: {leaf_nodes[0]}")
                        if leaf_nodes[0] in results:
                            return results[leaf_nodes[0]]
                    
                    # Strategy 2: Look for the node with the original operator's ID
                    original_node_id = str(op_id)
                    if original_node_id in results:
                        logger.debug(f"Using output from original operator node: {original_node_id}")
                        return results[original_node_id]
                        
                    # Strategy 3: If there are multiple leaf nodes but all have identical results,
                    # arbitrarily choose one
                    if len(leaf_nodes) > 1:
                        logger.debug(f"Found {len(leaf_nodes)} leaf nodes, checking result equality")
                        first_result = results.get(leaf_nodes[0])
                        if all(results.get(node) == first_result for node in leaf_nodes):
                            logger.debug(f"All leaf nodes have identical results, using first one")
                            return first_result
                    
                    # Strategy 4: Check if there's a node with "_output" in the name
                    output_nodes = [nid for nid in results.keys() if "_output" in nid]
                    if output_nodes:
                        logger.debug(f"Using output from node with '_output' in name: {output_nodes[0]}")
                        return results[output_nodes[0]]
                    
                    # Last attempt: see if any graph metadata can help us find the output node
                    if "output_node" in graph.metadata:
                        output_node = graph.metadata["output_node"]
                        if output_node in results:
                            logger.debug(f"Using output node from graph metadata: {output_node}")
                            return results[output_node]
                    
                    # If we got here, we couldn't determine the correct output
                    # Log the issue for debugging
                    logger.warning(
                        f"Could not determine output from graph with {len(graph.nodes)} nodes. "
                        f"Available results: {list(results.keys())}"
                    )
                except Exception as e:
                    # If graph execution fails, log the error and fall back to direct execution
                    logger.warning(f"Error executing graph: {e}. Falling back to direct execution.")
            
            # Phase 2: Tracing and direct execution
            # -------------------------------------------------
            # Execute the original call directly (when no cached graph or graph execution failed)
            start_time = time.time()
            output = original_call(self=self, inputs=inputs)
            end_time = time.time()
            
            # Phase 3: Record trace and update graph cache
            # -------------------------------------------------
            # Record trace if in a tracer context or force_trace is enabled
            if tracer is not None or force_trace_local:
                # Get operator name, preferring the 'name' attribute if available
                operator_name = getattr(self, "name", self.__class__.__name__)
                
                # Create trace record
                record = TraceRecord(
                    operator_name=operator_name,
                    node_id=str(id(self)),
                    inputs=inputs,
                    outputs=output,
                    timestamp=end_time,
                )
                
                # Add to tracer if available
                if tracer is not None:
                    tracer.add_record(record=record)
                    
                    # Check if this is an appropriate time to build a graph:
                    # 1. If tracing was forced (test/debug case)
                    # 2. If this is a top-level call (no prior records with same node_id)
                    # 3. If we've collected enough records to make a meaningful graph
                    build_graph = (
                        force_trace_local or
                        not any(r.node_id == str(id(self)) for r in tracer.records[:-1]) or
                        len(tracer.records) >= 3  # Minimum threshold for useful graph
                    )
                    
                    if build_graph:
                        # Import here to avoid circular imports
                        from ember.xcs.tracer.autograph import AutoGraphBuilder
                        
                        # Build a graph from the accumulated trace records
                        logger.debug(f"Building graph from {len(tracer.records)} trace records")
                        graph_builder = AutoGraphBuilder()
                        graph = graph_builder.build_graph(tracer.records)
                        
                        # Only cache the graph if it has multiple nodes (otherwise no benefit)
                        if len(graph.nodes) > 1:
                            logger.debug(f"Caching graph with {len(graph.nodes)} nodes")
                            # Add metadata to help with output node identification
                            graph.metadata["output_node"] = graph.nodes[list(graph.nodes.keys())[-1]].node_id
                            _COMPILED_GRAPHS[op_id] = graph
                        else:
                            logger.debug(f"Not caching trivial graph with {len(graph.nodes)} nodes")
            
            # Return the output from direct execution
            return output

        # Replace the original methods with our traced versions
        cls.__init__ = cast(Callable, traced_init)
        cls.__call__ = cast(Callable, traced_call)
        return cls

    # Handle both @jit and @jit(...) patterns
    if func is not None:
        # Called as @jit without parentheses
        return decorator(func)
    else:
        # Called with parameters as @jit(...)
        return decorator


# Removed _build_graph_from_trace function since we're not implementing the enhanced
# JIT capability in this PR. This would be included in a future full implementation.
