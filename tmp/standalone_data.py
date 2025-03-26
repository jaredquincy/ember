"""Standalone data utilities for Ember.

This module implements a standalone version of Ember's data utilities that can be used
without relying on the full Ember framework.
"""

from enum import Enum
from typing import Any, Dict, List
import random

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Enumeration of dataset task types."""

    MULTIPLE_CHOICE = "multiple_choice"
    BINARY_CLASSIFICATION = "binary_classification"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"


class DatasetInfo(BaseModel):
    """Model representing essential dataset information."""

    name: str
    description: str
    source: str
    task_type: TaskType


class DatasetEntry(BaseModel):
    """Model for a single dataset entry."""

    query: str
    choices: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dataset:
    """Simple dataset container."""

    def __init__(self, entries: List[DatasetEntry], info: DatasetInfo = None):
        self.entries = entries
        self.info = info

    def __getitem__(self, idx):
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)

    def sample(self, n: int = 1):
        """Sample n entries from the dataset."""
        return Dataset(
            entries=random.sample(self.entries, min(n, len(self.entries))),
            info=self.info,
        )


class DatasetBuilder:
    """Builder for creating datasets."""

    def __init__(self):
        self._entries = []
        self._info = None
        self._seed = 42

    def add_entry(
        self, query: str, choices: Dict[str, str], metadata: Dict[str, Any] = None
    ):
        """Add an entry to the dataset."""
        self._entries.append(
            DatasetEntry(query=query, choices=choices, metadata=metadata or {})
        )
        return self

    def info(self, name: str, description: str, source: str, task_type: TaskType):
        """Set dataset info."""
        self._info = DatasetInfo(
            name=name, description=description, source=source, task_type=task_type
        )
        return self

    def seed(self, seed_value: int):
        """Set random seed for sampling."""
        self._seed = seed_value
        random.seed(seed_value)
        return self

    def build(self):
        """Build the dataset."""
        return Dataset(entries=self._entries, info=self._info)


def create_example_dataset():
    """Create an example multiple-choice dataset."""
    builder = DatasetBuilder()

    # Set dataset info
    builder.info(
        name="example_dataset",
        description="An example multiple-choice dataset",
        source="ember.examples",
        task_type=TaskType.MULTIPLE_CHOICE,
    )

    # Add some example entries
    builder.add_entry(
        query="What is the capital of France?",
        choices={"A": "Berlin", "B": "Madrid", "C": "Paris", "D": "Rome"},
        metadata={"correct_answer": "C", "category": "Geography"},
    )

    builder.add_entry(
        query="Which of these is a mammal?",
        choices={"A": "Shark", "B": "Snake", "C": "Eagle", "D": "Dolphin"},
        metadata={"correct_answer": "D", "category": "Biology"},
    )

    builder.add_entry(
        query="What is the square root of 144?",
        choices={"A": "10", "B": "12", "C": "14", "D": "16"},
        metadata={"correct_answer": "B", "category": "Mathematics"},
    )

    return builder.build()


def main():
    """Run the standalone data example."""
    print("Ember Standalone Data Example")
    print("============================")

    # Create a dataset
    dataset = create_example_dataset()
    print(f"Created dataset with {len(dataset)} entries")
    print(f"Dataset info: {dataset.info}")

    # Display each entry
    for i, entry in enumerate(dataset):
        print(f"\nEntry {i+1}:")
        print(f"Question: {entry.query}")
        print(f"Choices:")
        correct = entry.metadata.get("correct_answer", "")
        for letter, text in entry.choices.items():
            is_correct = "✓" if letter == correct else " "
            print(f"  {letter}. {text} {is_correct}")
        print(f"Category: {entry.metadata.get('category', 'Unknown')}")

    # Demonstrate sampling
    print("\nSampling Example:")
    sample = dataset.sample(2)
    print(f"Sampled {len(sample)} entries: {[e.query for e in sample]}")


if __name__ == "__main__":
    main()
