"""
Test XCS Implementation

This example demonstrates the use of the XCS (eXecutable Computation System) API
directly. It shows how to use JIT compilation, vectorization, and automatic 
graph building for high-performance operator execution.

To run:
    poetry run python src/ember/examples/test_xcs_implementation.py
"""

import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

# Import the XCS module directly
project_root = Path(__file__).parent.parent.parent.parent
xcs_path = project_root / "src" / "ember" / "xcs" / "__init__.py"
spec = importlib.util.spec_from_file_location("ember.xcs", xcs_path)
xcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xcs)

# Now we can use the core XCS functionality
from functools import partial
import time

def main():
    """Test the XCS implementation with examples."""
    print("\n=== Testing XCS Implementation ===\n")
    
    # Test JIT decorator
    print("Testing JIT Compilation:")
    
    @xcs.jit
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers (JIT compiled)."""
        return a * b
    
    # Measure execution time
    start = time.time()
    result = multiply(10, 20)
    duration = time.time() - start
    
    print(f"  Result of multiply(10, 20): {result}")
    print(f"  Execution time: {duration:.6f} seconds")
    print("  (First call includes compilation overhead)")
    
    # Test vmap for vectorization
    print("\nTesting Vectorized Mapping (vmap):")
    
    def square(x: int) -> int:
        """Square a number."""
        return x * x
    
    vectorized_square = xcs.vmap(square)
    input_list = [1, 2, 3, 4, 5]
    result = vectorized_square(input_list)
    
    print(f"  Input: {input_list}")
    print(f"  Vectorized square: {result}")
    
    # Test pmap for parallel execution
    print("\nTesting Parallel Mapping (pmap):")
    
    def slow_operation(x: int) -> int:
        """A slow operation that simulates computational work."""
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    # Sequential execution for comparison
    start = time.time()
    sequential_results = [slow_operation(x) for x in range(10)]
    sequential_time = time.time() - start
    
    # Parallel execution
    parallel_op = xcs.pmap(slow_operation)
    start = time.time()
    parallel_results = parallel_op(list(range(10)))
    parallel_time = time.time() - start
    
    print(f"  Sequential execution time: {sequential_time:.6f} seconds")
    print(f"  Parallel execution time: {parallel_time:.6f} seconds")
    print(f"  Speed improvement: {sequential_time / parallel_time:.2f}x")
    
    # Test autograph for automatic graph building
    print("\nTesting Automatic Graph Building (autograph):")
    
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
    
    with xcs.autograph() as graph:
        # These operations are recorded, not executed immediately
        sum_result = add(5, 3)  # node1
        product = multiply(sum_result, 2)  # node2
    
    print("  Graph built successfully")
    print("  Executing graph...")
    
    # Execute the graph
    results = xcs.execute(graph)
    print(f"  Graph execution results: {results}")
    
    print("\nXCS Implementation Test Complete!")

if __name__ == "__main__":
    main()