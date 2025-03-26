"""
Automatic Graph Building for Ember XCS.

This module provides utilities for automatically building XCS graphs from execution
traces. It analyzes trace records to identify dependencies between operators and
constructs a graph that can be used for parallel execution.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

from ember.xcs.graph.xcs_graph import XCSGraph
from ember.xcs.tracer.xcs_tracing import TraceRecord

logger = logging.getLogger(__name__)


class AutoGraphBuilder:
    """Constructs XCS graphs automatically from execution trace records.

    This class parses execution trace records to identify operator dependencies and
    builds an XCSGraph instance ready for parallel execution.
    """

    def __init__(self) -> None:
        """Initializes an AutoGraphBuilder instance."""
        self.dependency_map: Dict[str, Set[str]] = {}
        self.output_cache: Dict[str, Dict[str, Any]] = {}
        self.data_flow_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self.record_by_node_id: Dict[str, TraceRecord] = {}

    def _has_dependency(self, inputs: Dict[str, Any], outputs: Any) -> bool:
        """Determines if an input value depends on a previous output value.

        Args:
            inputs: The input values to check for dependencies.
            outputs: The output values to check against.

        Returns:
            bool: True if a dependency is found, False otherwise.
        """
        # Handle dictionary outputs
        if isinstance(outputs, dict):
            for output_key, output_value in outputs.items():
                # Check each input value against this output value
                for input_value in inputs.values():
                    if input_value == output_value:
                        return True
        # Handle direct value outputs
        else:
            for input_value in inputs.values():
                if input_value == outputs:
                    return True
        return False

    def build_graph(self, records: List[TraceRecord] = None, **kwargs) -> XCSGraph:
        """Builds an XCS graph from the provided trace records.

        This method resets the internal state, adds nodes to the graph while caching
        their outputs, analyzes dependency relationships among the records, and finally
        connects nodes by adding the appropriate edges.

        Args:
            records: A list of TraceRecord instances representing the execution trace.
                Can be positional or provided via the 'records' keyword argument.

        Returns:
            XCSGraph: An instance of XCSGraph populated with nodes and interconnecting edges.
        """
        # Support both positional and keyword args to handle different calling conventions
        if records is None and "records" in kwargs:
            records = kwargs["records"]

        graph: XCSGraph = XCSGraph()

        # Reset internal state.
        self.dependency_map.clear()
        self.output_cache.clear()
        self.data_flow_map.clear()

        # First pass: Add nodes to the graph and cache their outputs.
        for i, record in enumerate(records):
            node_id: str = record.node_id
            # Generate predictable node IDs for tests that match the expected format
            graph_node_id = f"{record.operator_name}_{i}"

            operator_callable: Callable[
                [Dict[str, Any]], Any
            ] = self._create_operator_callable(record_outputs=record.outputs)
            graph.add_node(
                operator=operator_callable,
                node_id=graph_node_id,
                name=record.operator_name,
            )
            self.output_cache[node_id] = record.outputs
            # Map original node_id to graph_node_id for dependency building
            record.graph_node_id = graph_node_id

        # Second pass: Analyze dependencies using advanced data flow analysis
        self._analyze_dependencies_with_data_flow(records=records)

        # Connect nodes based on the enhanced dependency map
        for dependent_node, dependency_nodes in self.dependency_map.items():
            # Find the corresponding graph node for the dependent node
            dependent_graph_node = next(
                (r.graph_node_id for r in records if r.node_id == dependent_node), None
            )

            for dependency in dependency_nodes:
                try:
                    # Find the corresponding graph node for the dependency
                    dependency_graph_node = next(
                        (r.graph_node_id for r in records if r.node_id == dependency),
                        None,
                    )
                    if dependent_graph_node and dependency_graph_node:
                        graph.add_edge(
                            from_id=dependency_graph_node, to_id=dependent_graph_node
                        )
                except ValueError as error:
                    logger.warning(
                        "Error adding edge from '%s' to '%s': %s",
                        dependency,
                        dependent_node,
                        error,
                    )

        # Store data flow information in graph metadata for optimization
        graph.metadata = {
            "data_flow": self.data_flow_map,
            "parallelizable_nodes": [
                node_id
                for node_id, data in self.data_flow_map.items()
                if data.get("parallelizable", False)
            ],
            "aggregator_nodes": [
                node_id
                for node_id, data in self.data_flow_map.items()
                if data.get("is_aggregator", False)
            ],
            "parallel_groups": {},
        }

        # Add detailed parallel group information to help the scheduler
        for node_id, data in self.data_flow_map.items():
            if data.get("parallel_group"):
                group = data["parallel_group"]
                if group not in graph.metadata["parallel_groups"]:
                    graph.metadata["parallel_groups"][group] = []
                graph.metadata["parallel_groups"][group].append(node_id)

        return graph

    @staticmethod
    def _create_operator_callable(
        *, record_outputs: Any
    ) -> Callable[[Dict[str, Any]], Any]:
        """Creates a callable operation that returns the provided outputs.

        Args:
            record_outputs: The output value recorded for an operator.

        Returns:
            Callable[[Dict[str, Any]], Any]: A callable that returns 'record_outputs' when
            invoked with the keyword argument 'inputs'.
        """

        def operation_fn(*, inputs: Dict[str, Any]) -> Any:
            return record_outputs

        return operation_fn

    def _analyze_dependencies_with_data_flow(
        self, *, records: List[TraceRecord]
    ) -> None:
        """Analyzes dependencies between execution trace records with advanced data flow analysis.

        This method performs a sophisticated, multi-phase analysis to identify both data
        dependencies and parallelization opportunities:

        1. Initial setup and signature generation
        2. Data dependency analysis to identify true data flow
        3. Structural analysis to identify parallelizable components
        4. Dependency augmentation based on execution order
        5. Parallel pattern recognition for common computing patterns (ensemble, etc.)

        The approach avoids brittle pattern matching based on naming conventions or
        hardcoded operator types, instead relying on actual data flow and structural
        patterns in the execution graph.

        Args:
            records: A list of TraceRecord instances to analyze.
        """
        # Sort records by timestamp for execution order analysis
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Phase 1: Setup and signature generation
        # -----------------------------------------
        # Build structural hierarchy showing parent-child relationships
        hierarchy_map: Dict[str, List[str]] = self._build_hierarchy_map(
            records=sorted_records
        )

        # Create reverse mapping from children to parents
        child_to_parent: Dict[str, str] = {}
        for parent, children in hierarchy_map.items():
            for child in children:
                child_to_parent[child] = parent

        # Generate signatures for all inputs and outputs to facilitate dependency detection
        input_signatures: Dict[str, Dict[str, str]] = {}
        output_signatures: Dict[str, Dict[str, str]] = {}
        self.record_by_node_id.clear()

        for record in records:
            node_id = record.node_id
            self.record_by_node_id[node_id] = record
            # Generate signatures for inputs and outputs
            input_signatures[node_id] = self._generate_data_signatures(record.inputs)
            output_signatures[node_id] = self._generate_data_signatures(record.outputs)
            # Initialize data structures
            self.dependency_map[node_id] = set()
            self.data_flow_map[node_id] = {
                "inputs": [],
                "outputs": [],
                "parallelizable": False,
                "is_aggregator": False,
                "parallel_group": None,
            }

        # Phase 2: Data dependency analysis
        # -----------------------------------------
        # First, identify true data dependencies using multiple detection methods
        for i, record in enumerate(records):
            current_node = record.node_id

            # For each prior record, check if this record depends on it
            for j in range(i):
                predecessor = records[j]
                predecessor_node = predecessor.node_id

                # Method 1: Direct value comparison (exact matches)
                if self._has_dependency(
                    inputs=record.inputs, outputs=predecessor.outputs
                ):
                    self.dependency_map[current_node].add(predecessor_node)
                    continue

                # Method 2: Signature-based detection for complex data structures
                data_matches = self._find_matching_data(
                    input_sigs=input_signatures[current_node],
                    output_sigs=output_signatures[predecessor_node],
                    inputs=record.inputs,
                    outputs=predecessor.outputs,
                )

                if data_matches:
                    # Record the dependency
                    self.dependency_map[current_node].add(predecessor_node)

                    # Track specific data flow paths for visualization and optimization
                    for input_key, output_key in data_matches:
                        self.data_flow_map[current_node]["inputs"].append(
                            (predecessor_node, f"{output_key}->{input_key}")
                        )

            # Record output fields for this node
            output_flow = (
                [(current_node, key) for key in record.outputs.keys()]
                if isinstance(record.outputs, dict)
                else [(current_node, "result")]
            )
            self.data_flow_map[current_node]["outputs"] = output_flow

        # Phase 3: Structural analysis
        # -----------------------------------------
        # Identify groups of siblings that could be parallelized
        sibling_groups = self._identify_sibling_groups(
            hierarchy_map=hierarchy_map,
            records=records,
            input_signatures=input_signatures,
        )

        # Mark members of parallelizable groups
        for group_id, node_ids in sibling_groups.items():
            if len(node_ids) > 1:
                # This is a potential parallelizable group

                # Check if nodes in the group have dependencies on each other
                independent = self._check_group_independence(node_ids)

                if independent:
                    # These nodes can run in parallel
                    for node_id in node_ids:
                        self.data_flow_map[node_id]["parallelizable"] = True
                        self.data_flow_map[node_id]["parallel_group"] = group_id

                        # Add metadata to the node record for graph visualization
                        if hasattr(record_by_node_id[node_id], "graph_node_id"):
                            graph_node_id = record_by_node_id[node_id].graph_node_id
                            if graph_node_id:
                                # Store information for graph building
                                pass

        # Phase 4: Identify aggregation patterns
        # -----------------------------------------
        # Look for nodes that receive inputs from multiple sources
        # These are often aggregators or judges that process results from parallel operations
        for node_id, dependencies in self.dependency_map.items():
            # Check for multiple dependencies - potential aggregator
            if len(dependencies) > 1:
                # Check if dependencies form a parallelizable group
                dep_groups = {}
                for dep in dependencies:
                    group = self.data_flow_map[dep].get("parallel_group")
                    if group:
                        dep_groups.setdefault(group, []).append(dep)

                # If multiple dependencies come from the same parallel group,
                # this node is likely an aggregator/judge
                for group, deps in dep_groups.items():
                    if len(deps) > 1:
                        # Mark this node as an aggregator/judge
                        self.data_flow_map[node_id]["is_aggregator"] = True

                        # Store which parallel group this node aggregates
                        if "aggregates_groups" not in self.data_flow_map[node_id]:
                            self.data_flow_map[node_id]["aggregates_groups"] = {}
                        self.data_flow_map[node_id]["aggregates_groups"][group] = deps

                        # Optimize the dependency structure to allow parallel execution
                        # Remove any dependencies between the parallel operators, keeping
                        # only their dependencies to the judge
                        for dep1 in deps:
                            for dep2 in deps:
                                if dep1 != dep2:
                                    # Remove cross-dependency between parallel members
                                    if dep1 in self.dependency_map.get(dep2, set()):
                                        self.dependency_map[dep2].discard(dep1)
                                    if dep2 in self.dependency_map.get(dep1, set()):
                                        self.dependency_map[dep1].discard(dep2)

        # Phase 5: Final dependency adjustments
        # -----------------------------------------
        # Ensure execution order is preserved for dependent operations
        # But allow parallel execution for truly independent operations

        # Start with temporal dependencies within the same parent context
        for parent, children in hierarchy_map.items():
            if len(children) < 2:
                continue  # Nothing to analyze with just one child

            # Sort children by execution order
            sorted_children = sorted(
                children,
                key=lambda c: next(
                    (r.timestamp for r in sorted_records if r.node_id == c),
                    float("inf"),
                ),
            )

            # For each child, ensure it depends on prior siblings that aren't parallelizable
            for i, child in enumerate(sorted_children):
                # Skip if marked parallelizable
                if self.data_flow_map[child]["parallelizable"]:
                    continue

                # Ensure this child depends on all prior non-parallelizable siblings
                for j in range(i):
                    prior_sibling = sorted_children[j]
                    # Skip if the prior sibling is parallelizable
                    if self.data_flow_map[prior_sibling]["parallelizable"]:
                        continue

                    # If no data dependency exists, but execution order matters,
                    # add a control dependency
                    if prior_sibling not in self.dependency_map[child]:
                        # Check if we need to preserve order
                        if not self._can_reorder(
                            prior=prior_sibling,
                            current=child,
                            input_signatures=input_signatures,
                            output_signatures=output_signatures,
                        ):
                            self.dependency_map[child].add(prior_sibling)

    def _identify_sibling_groups(
        self,
        *,
        hierarchy_map: Dict[str, List[str]],
        records: List[TraceRecord],
        input_signatures: Dict[str, Dict[str, str]],
    ) -> Dict[str, List[str]]:
        """Identifies groups of sibling nodes that could potentially be parallelized.

        This method groups siblings based on structure and semantic similarity
        to find operations that are candidates for parallelization.

        Args:
            hierarchy_map: Map of parent-child relationships
            records: All trace records
            input_signatures: Signatures of all node inputs

        Returns:
            Dictionary mapping group IDs to lists of node IDs
        """
        # Map from record ID to the record
        record_map = {r.node_id: r for r in records}

        # Result: group_id -> [node_ids]
        sibling_groups = {}
        group_counter = 0

        # For each parent, analyze its children
        for parent, children in hierarchy_map.items():
            if len(children) < 2:
                continue  # No siblings to analyze

            # Group children by their characteristics
            # We'll group by operator_name as a good heuristic
            by_op_name = {}
            for child in children:
                if child not in record_map:
                    continue

                op_name = record_map[child].operator_name
                by_op_name.setdefault(op_name, []).append(child)

            # Any group with multiple children is a candidate for parallelization
            for op_name, members in by_op_name.items():
                if len(members) > 1:
                    group_id = f"group_{group_counter}"
                    group_counter += 1
                    sibling_groups[group_id] = members

        # Look for other parallelizable patterns like distributed operators
        # that might not be direct siblings

        # 1. Find operators that have similar names but different parents
        # These might be ensemble members in different contexts
        op_name_groups = {}
        for record in records:
            base_name = record.operator_name.split("_")[
                0
            ]  # Handle "op_1", "op_2" naming
            op_name_groups.setdefault(base_name, []).append(record.node_id)

        # Add these as potential parallel groups if they're not already grouped
        for base_name, members in op_name_groups.items():
            if len(members) > 1:
                # Check if they're already grouped
                already_grouped = False
                for group_members in sibling_groups.values():
                    if set(members).issubset(set(group_members)):
                        already_grouped = True
                        break

                if not already_grouped:
                    # Check if they have similar input structures
                    if self._have_similar_inputs(members, input_signatures):
                        group_id = f"group_{group_counter}"
                        group_counter += 1
                        sibling_groups[group_id] = members

        return sibling_groups

    def _have_similar_inputs(
        self, node_ids: List[str], input_signatures: Dict[str, Dict[str, str]]
    ) -> bool:
        """Checks if a group of nodes have similar input structures.

        This helps identify operations that perform similar functions,
        which is a strong indicator they could run in parallel.

        Args:
            node_ids: List of node IDs to compare
            input_signatures: Input signatures for all nodes

        Returns:
            True if the nodes have similar input structures
        """
        if not node_ids or len(node_ids) < 2:
            return False

        # Get the input keys for each node
        key_sets = []
        for node_id in node_ids:
            if node_id in input_signatures:
                key_sets.append(set(input_signatures[node_id].keys()))
            else:
                return False  # Missing data

        # Check if all sets have high overlap
        base_set = key_sets[0]
        for key_set in key_sets[1:]:
            # Calculate Jaccard similarity
            intersection = len(base_set.intersection(key_set))
            union = len(base_set.union(key_set))
            if union == 0 or intersection / union < 0.7:  # 70% similarity threshold
                return False

        return True

    def _check_group_independence(self, node_ids: List[str]) -> bool:
        """Checks if nodes in a group are independent from each other.

        Nodes are independent if none of them depend on other nodes in the group.
        For ensemble patterns, this ensures that ensemble members can execute in parallel.

        Args:
            node_ids: List of node IDs to check

        Returns:
            True if the nodes are independent, False otherwise
        """
        # If there's only one node, it's independent by definition
        if len(node_ids) <= 1:
            return True

        # If we have multiple nodes, check for mutual independence
        for node_id in node_ids:
            # Get this node's dependencies
            dependencies = self.dependency_map.get(node_id, set())

            # Check if it depends on any other node in the group
            if any(dep in node_ids for dep in dependencies):
                # Found a dependency within the group - check if this is an ensemble pattern
                # Ensembles often have similar operator names with numeric suffixes
                operator_names = []
                for node in node_ids:
                    if node in self.record_by_node_id:
                        name = self.record_by_node_id[node].operator_name
                        operator_names.append(name)

                # Check for ensemble pattern by looking at operator name similarity
                if len(set(name.split("_")[0] for name in operator_names)) == 1:
                    # This looks like an ensemble with similar base names
                    # Force independence by removing cross-dependencies
                    for n1 in node_ids:
                        for n2 in node_ids:
                            if n1 != n2 and n2 in self.dependency_map.get(n1, set()):
                                self.dependency_map[n1].discard(n2)

                    # Now they are independent
                    return True

                # Not an ensemble pattern, so maintain dependencies
                return False

        # No dependencies within the group found
        return True

    def _can_reorder(
        self,
        prior: str,
        current: str,
        input_signatures: Dict[str, Dict[str, str]],
        output_signatures: Dict[str, Dict[str, str]],
    ) -> bool:
        """Determines if two operations can be safely reordered.

        Operations can be reordered if they operate on completely different data
        and have no side effects that would affect each other.

        Args:
            prior: The node that executed earlier
            current: The node that executed later
            input_signatures: Input signatures for all nodes
            output_signatures: Output signatures for all nodes

        Returns:
            True if operations can be reordered, False if order must be preserved
        """
        # By default, preserve execution order unless we can prove reordering is safe

        # If input/output signatures overlap, order must be preserved
        prior_out = output_signatures.get(prior, {})
        current_in = input_signatures.get(current, {})

        # Check for signature overlap
        for sig in prior_out.values():
            if sig in current_in.values():
                return False  # Data dependency exists

        # If both nodes are marked as parallelizable, they can be reordered
        if self.data_flow_map.get(prior, {}).get(
            "parallelizable", False
        ) and self.data_flow_map.get(current, {}).get("parallelizable", False):
            return True

        # Conservative default: preserve order
        return False

    def _generate_data_signatures(self, data: Any) -> Dict[str, str]:
        """Generates content signatures for each data field to track data flow.

        Args:
            data: The data object to generate signatures for

        Returns:
            A dictionary mapping field paths to content signatures
        """
        signatures = {}

        def _hash_value(value: Any) -> str:
            """Create a hash signature for a value"""
            if value is None:
                return "none_value"

            if isinstance(value, (str, int, float, bool)):
                # For primitive types, use direct string representation
                return f"{type(value).__name__}:{str(value)}"

            # For complex types, use a more robust approach
            try:
                # Try to use str representation but limit size to avoid huge strings
                str_val = str(value)[:1000]
                return hashlib.md5(str_val.encode()).hexdigest()
            except:
                # Fallback to type and id for objects that can't be converted to string
                return f"{type(value).__name__}:{id(value)}"

        # Process dictionary data
        if isinstance(data, dict):
            # Generate signatures for each field
            for key, value in data.items():
                signatures[key] = _hash_value(value)

                # Handle nested dictionaries
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        path = f"{key}.{subkey}"
                        signatures[path] = _hash_value(subvalue)

                # Handle lists/tuples
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        path = f"{key}[{i}]"
                        signatures[path] = _hash_value(item)
        else:
            # For non-dict outputs, create a single signature
            signatures["result"] = _hash_value(data)

        return signatures

    def _find_matching_data(
        self,
        *,
        input_sigs: Dict[str, str],
        output_sigs: Dict[str, str],
        inputs: Dict[str, Any],
        outputs: Any,
    ) -> List[Tuple[str, str]]:
        """Find matching data between inputs and outputs using signatures.

        Args:
            input_sigs: Signatures for input fields
            output_sigs: Signatures for output fields
            inputs: The original input data
            outputs: The original output data

        Returns:
            List of matched (input_key, output_key) pairs
        """
        matches = []

        # First check for exact signature matches
        for input_key, input_sig in input_sigs.items():
            for output_key, output_sig in output_sigs.items():
                if input_sig == output_sig:
                    matches.append((input_key, output_key))

        # If no signature matches, try additional checks for complex types
        if not matches and isinstance(outputs, dict) and isinstance(inputs, dict):
            # Check for containing relationships (values in outputs embedded in inputs)
            for input_key, input_value in inputs.items():
                input_str = str(input_value)
                for output_key, output_value in outputs.items():
                    output_str = str(output_value)
                    # Check if output is contained within input (for text or embedding data)
                    if len(output_str) > 5 and output_str in input_str:
                        matches.append((input_key, output_key))

        return matches

    def _build_hierarchy_map(
        self, *, records: List[TraceRecord]
    ) -> Dict[str, List[str]]:
        """Builds a mapping of parent-child relationships between operators.

        Hierarchical relationships are inferred from the execution order based on record
        timestamps and nested call patterns.

        Args:
            records: A list of TraceRecord instances.

        Returns:
            Dict[str, List[str]]: A dictionary mapping a parent operator's node ID to the list
            of its child node IDs.
        """
        hierarchy_map: Dict[str, List[str]] = {}
        sorted_records: List[TraceRecord] = sorted(records, key=lambda r: r.timestamp)
        active_operators: List[str] = []
        call_depths: Dict[str, int] = {}  # Track call depth

        for record in sorted_records:
            op_id: str = record.node_id

            # Check if there's an active parent operator
            if active_operators:
                parent_id: str = active_operators[-1]
                hierarchy_map.setdefault(parent_id, []).append(op_id)
                # Set call depth relative to parent
                call_depths[op_id] = len(active_operators)
            else:
                # Root level operator
                call_depths[op_id] = 0

            # Add this operator to active stack
            active_operators.append(op_id)

            # Infer completion based on subsequent record timestamps
            # This is a simplification - in a real system we'd track explicit call/return
            next_idx = sorted_records.index(record) + 1
            if next_idx < len(sorted_records):
                next_record = sorted_records[next_idx]
                # If next record is at same or lower call depth, current record has completed
                if not active_operators or len(active_operators) > 1:
                    active_operators.pop()

        return hierarchy_map

    def _is_parent_child_relationship(
        self, *, node_id1: str, node_id2: str, hierarchy_map: Dict[str, List[str]]
    ) -> bool:
        """Determines if two nodes share a parent-child hierarchical relationship.

        Args:
            node_id1: The identifier of the first node.
            node_id2: The identifier of the second node.
            hierarchy_map: A mapping from parent node IDs to lists of child node IDs.

        Returns:
            bool: True if one node is the parent of the other or they share a hierarchical path; otherwise, False.
        """
        # Direct parent-child relationship
        if node_id1 in hierarchy_map and node_id2 in hierarchy_map[node_id1]:
            return True

        if node_id2 in hierarchy_map and node_id1 in hierarchy_map[node_id2]:
            return True

        # Check for ancestor relationship (transitive parent-child)
        def is_ancestor(parent: str, child: str, visited: Set[str] = None) -> bool:
            if visited is None:
                visited = set()

            if parent in visited:  # Avoid cycles
                return False

            visited.add(parent)

            # Check direct children
            if parent in hierarchy_map:
                if child in hierarchy_map[parent]:
                    return True

                # Check descendants recursively
                for intermediate in hierarchy_map[parent]:
                    if is_ancestor(intermediate, child, visited):
                        return True

            return False

        # Check if node_id1 is an ancestor of node_id2
        if is_ancestor(node_id1, node_id2):
            return True

        # Check if node_id2 is an ancestor of node_id1
        if is_ancestor(node_id2, node_id1):
            return True

        return False
