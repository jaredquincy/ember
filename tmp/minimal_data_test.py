"""Minimal test script for data functionality.

This script creates a very simple dataset using the base components from the data module
without depending on the full Ember API, avoiding the circular imports.
"""

import sys
import os
from typing import List, Dict, Any

# Import only the base components we need
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType


def main():
    """Run a minimal test of the data functionality."""
    print("Testing minimal data functionality...")

    # Create a sample dataset
    items = []
    for i in range(3):
        entry = DatasetEntry(
            query=f"Question {i}",
            choices={
                "A": f"Option A for question {i}",
                "B": f"Option B for question {i}",
                "C": f"Option C for question {i}",
                "D": f"Option D for question {i}",
            },
            metadata={"correct_answer": "B", "category": "Test"},
        )
        items.append(entry)

    # Print the dataset entries
    print(f"Created {len(items)} dataset entries")

    # Display the first entry
    print("\nSample entry:")
    print(f"Query: {items[0].query}")
    print(f"Choices: {items[0].choices}")
    print(f"Metadata: {items[0].metadata}")

    # Create a dataset info
    info = DatasetInfo(
        name="test_dataset",
        description="A test dataset",
        source="local",
        task_type=TaskType.MULTIPLE_CHOICE,
    )
    print(f"\nDataset info: {info}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
