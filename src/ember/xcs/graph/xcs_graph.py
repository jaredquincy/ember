"""Computation graph for XCS execution.

Defines a directed acyclic graph structure for representing and executing
computational flows. Operators form nodes in the graph, with edges representing
data dependencies between operations.

Example:
    ```python
    graph = XCSGraph()
    
    # Add computation nodes
    input_node = graph.add_node(preprocess_fn, name="preprocess")
    compute_node = graph.add_node(compute_fn, name="compute")
    output_node = graph.add_node(postprocess_fn, name="postprocess")
    
    # Define data flow
    graph.add_edge(input_node, compute_node)
    graph.add_edge(compute_node, output_node)
    
    # Execute the computation with an execution engine
    from ember.xcs.engine import execute
    results = execute(graph, inputs={"data": input_data})
    ```
"""

import dataclasses
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast


@dataclasses.dataclass
class XCSNode:
    """Single computation node in an execution graph.

    Represents one operation in a computational flow with its connections
    to other nodes. Each node contains an executable operator and maintains
    its position in the graph through edge lists.

    Attributes:
        operator: Callable function or operator executing this node's computation
        node_id: Unique identifier for addressing this node in the graph
        inbound_edges: Node IDs that provide inputs to this node
        outbound_edges: Node IDs that consume output from this node
        name: Human-readable label for debugging and visualization
        metadata: Additional node properties (e.g., cost estimates, device placement)
    """

    operator: Callable[..., Dict[str, Any]]
    node_id: str
    inbound_edges: List[str] = dataclasses.field(default_factory=list)
    outbound_edges: List[str] = dataclasses.field(default_factory=list)
    name: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


# For backward compatibility
XCSGraphNode = XCSNode


class XCSGraph:
    """Directed graph for computational workflows.

    Provides a structure for defining complex computational flows as directed
    graphs. Supports operations needed for graph analysis, transformation, and
    execution by the XCS execution engine.
    """

    def __init__(self) -> None:
        """Creates an empty computation graph."""
        self.nodes: Dict[str, XCSNode] = {}
        self.metadata: Dict[str, Any] = {}

    def add_node(
        self,
        operator: Callable[..., Dict[str, Any]],
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adds a computation node to the graph.

        Args:
            operator: Function or operator to execute at this node
            node_id: Unique identifier (auto-generated if None)
            name: Human-readable label for the node
            metadata: Additional properties for analysis and optimization

        Returns:
            Generated or provided node ID

        Raises:
            ValueError: If node_id already exists in the graph
        """
        if node_id is None:
            node_id = str(uuid.uuid4())

        if node_id in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' already exists.")

        self.nodes[node_id] = XCSNode(
            operator=operator, node_id=node_id, name=name, metadata=metadata or {}
        )
        return node_id

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Creates a directed data dependency between nodes.

        Establishes that the output of one node flows into another,
        forming a directed edge in the computation graph.

        Args:
            from_id: Source node producing output data
            to_id: Destination node consuming the data

        Raises:
            ValueError: If either node doesn't exist in the graph
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node '{from_id}' does not exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Destination node '{to_id}' does not exist.")

        self.nodes[from_id].outbound_edges.append(to_id)
        self.nodes[to_id].inbound_edges.append(from_id)

    def topological_sort(self) -> List[str]:
        """Orders nodes so dependencies come before dependents.

        Produces an execution ordering where each node appears after
        all nodes it depends on, ensuring valid sequential execution.

        Returns:
            List of node IDs in dependency-respecting order

        Raises:
            ValueError: If graph contains cycles (not a DAG)
        """
        # Track remaining dependencies for each node
        in_degree = {
            node_id: len(node.inbound_edges) for node_id, node in self.nodes.items()
        }
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        sorted_nodes = []

        # Process nodes in topological order
        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)

            for neighbor in self.nodes[current].outbound_edges:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Verify complete ordering (no cycles)
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return sorted_nodes

    def __str__(self) -> str:
        """Creates a human-readable graph representation.

        Generates a structured text description showing nodes and
        their connections, useful for debugging and visualization.

        Returns:
            Multi-line string describing the graph structure
        """
        nodes_str = [
            f"Node {node_id}: {node.name or 'unnamed'}"
            for node_id, node in self.nodes.items()
        ]
        edges_str = []
        for node_id, node in self.nodes.items():
            for edge in node.outbound_edges:
                edges_str.append(f"{node_id} -> {edge}")

        return (
            f"XCSGraph with {len(self.nodes)} nodes:\n"
            + "\n".join(nodes_str)
            + "\n\nEdges:\n"
            + "\n".join(edges_str)
        )


def merge_xcs_graphs(base: XCSGraph, additional: XCSGraph, namespace: str) -> XCSGraph:
    """Combines two computation graphs with namespace isolation.

    Creates a new graph containing all nodes from both input graphs,
    with nodes from the additional graph prefixed to avoid collisions.
    Preserves all edge connections, adjusting IDs as needed.

    Args:
        base: Primary graph to merge into
        additional: Secondary graph to incorporate with namespace prefixing
        namespace: Prefix for additional graph's node IDs for isolation

    Returns:
        New graph containing nodes and edges from both inputs

    Example:
        ```python
        # Merge specialized processing graph into main workflow
        main_graph = XCSGraph()  # Main computation pipeline
        process_graph = XCSGraph()  # Specialized processing subgraph

        # Combine while isolating process_graph nodes
        merged = merge_xcs_graphs(main_graph, process_graph, "process")
        ```
    """
    merged = XCSGraph()

    # Copy base graph nodes with original IDs
    for node_id, node in base.nodes.items():
        merged.add_node(operator=node.operator, node_id=node_id, name=node.name)

    # Copy additional graph nodes with namespaced IDs to prevent collisions
    node_mapping = {}  # Maps original IDs to namespaced IDs
    for node_id, node in additional.nodes.items():
        namespaced_id = f"{namespace}_{node_id}"
        # Ensure uniqueness with random suffix if needed
        if namespaced_id in merged.nodes:
            namespaced_id = f"{namespace}_{node_id}_{uuid.uuid4().hex[:8]}"

        merged.add_node(operator=node.operator, node_id=namespaced_id, name=node.name)
        node_mapping[node_id] = namespaced_id

    # Recreate edge connections from base graph (unchanged)
    for node_id, node in base.nodes.items():
        for edge_to in node.outbound_edges:
            merged.add_edge(from_id=node_id, to_id=edge_to)

    # Recreate edge connections from additional graph (with ID translation)
    for node_id, node in additional.nodes.items():
        from_id = node_mapping[node_id]
        for edge_to in node.outbound_edges:
            # Translate destination ID if it's from additional graph
            to_id = node_mapping.get(edge_to, edge_to)
            merged.add_edge(from_id=from_id, to_id=to_id)

    return merged
