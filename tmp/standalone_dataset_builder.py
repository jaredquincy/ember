"""Standalone implementation of the DatasetBuilder pattern.

This script provides a working example of the DatasetBuilder pattern shown in the README,
using mock datasets that don't require actual registration.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
import random
from pydantic import BaseModel, Field

# Define the core types needed
T = TypeVar("T")


class TaskType:
    """Enumeration of dataset task types."""

    MULTIPLE_CHOICE = "multiple_choice"
    BINARY_CLASSIFICATION = "binary_classification"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"


class DatasetInfo(BaseModel):
    """Metadata for a dataset."""

    name: str
    description: str
    source: str
    task_type: str


class DatasetEntry(BaseModel):
    """Entry in a dataset."""

    query: str
    choices: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dataset(Generic[T]):
    """A dataset of entries."""

    def __init__(self, entries: List[T], info: Optional[DatasetInfo] = None):
        """Initialize the dataset.

        Args:
            entries: List of dataset entries
            info: Dataset metadata
        """
        self.entries = entries
        self.info = info

    def __getitem__(self, idx: int) -> T:
        """Get an entry by index.

        Args:
            idx: Index of the entry

        Returns:
            The entry at the specified index
        """
        return self.entries[idx]

    def __len__(self) -> int:
        """Get the number of entries.

        Returns:
            Number of entries in the dataset
        """
        return len(self.entries)

    def __iter__(self):
        """Iterate over entries.

        Returns:
            Iterator over dataset entries
        """
        return iter(self.entries)


# Create a mock dataset registry
class DatasetRegistry:
    """Registry of available datasets."""

    def __init__(self):
        """Initialize the registry."""
        self._datasets = {}
        self._register_defaults()

    def register(self, name: str, generator_fn: Callable[[], Dataset]):
        """Register a dataset.

        Args:
            name: Name of the dataset
            generator_fn: Function that generates the dataset
        """
        self._datasets[name] = generator_fn

    def get(self, name: str) -> Optional[Dataset]:
        """Get a dataset by name.

        Args:
            name: Name of the dataset

        Returns:
            The dataset if found, None otherwise
        """
        if name not in self._datasets:
            return None
        return self._datasets[name]()

    def list_datasets(self) -> List[str]:
        """List all registered datasets.

        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())

    def _register_defaults(self):
        """Register default datasets."""
        # Register MMLU dataset
        self.register("mmlu", self._create_mmlu_dataset)

        # Register other default datasets
        self.register("truthful_qa", self._create_truthful_qa_dataset)
        self.register("commonsense_qa", self._create_commonsense_qa_dataset)

    def _create_mmlu_dataset(self) -> Dataset:
        """Create a mock MMLU dataset.

        Returns:
            MMLU dataset
        """
        # Define subject categories
        subjects = {
            "physics": [
                (
                    "What is Newton's first law?",
                    {
                        "A": "Objects in motion stay in motion",
                        "B": "Force equals mass times acceleration",
                        "C": "Energy is conserved",
                        "D": "Matter cannot be created or destroyed",
                    },
                    "A",
                ),
                (
                    "What particle is exchanged in electromagnetic interactions?",
                    {"A": "Graviton", "B": "Photon", "C": "Gluon", "D": "W boson"},
                    "B",
                ),
            ],
            "mathematics": [
                (
                    "What is the derivative of sin(x)?",
                    {"A": "cos(x)", "B": "-sin(x)", "C": "tan(x)", "D": "-cos(x)"},
                    "A",
                ),
                (
                    "What is the value of π (pi) to two decimal places?",
                    {"A": "3.41", "B": "3.14", "C": "3.12", "D": "3.18"},
                    "B",
                ),
            ],
        }

        # Create entries
        entries = []
        for subject, questions in subjects.items():
            for i, (question, choices, answer) in enumerate(questions):
                entry = DatasetEntry(
                    query=question,
                    choices=choices,
                    metadata={
                        "id": f"mmlu_{subject}_{i}",
                        "subject": subject,
                        "correct_answer": answer,
                    },
                )
                entries.append(entry)

        # Create info
        info = DatasetInfo(
            name="mmlu",
            description="Massive Multitask Language Understanding",
            source="cais/mmlu",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        return Dataset(entries, info)

    def _create_truthful_qa_dataset(self) -> Dataset:
        """Create a mock TruthfulQA dataset.

        Returns:
            TruthfulQA dataset
        """
        # Create entries
        entries = []
        for i in range(5):
            entry = DatasetEntry(
                query=f"TruthfulQA question {i}?",
                choices={
                    "A": f"True answer {i}",
                    "B": f"False but plausible answer {i}",
                },
                metadata={"id": f"truthful_qa_{i}", "correct_answer": "A"},
            )
            entries.append(entry)

        # Create info
        info = DatasetInfo(
            name="truthful_qa",
            description="TruthfulQA dataset",
            source="truthful_qa",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        return Dataset(entries, info)

    def _create_commonsense_qa_dataset(self) -> Dataset:
        """Create a mock CommonsenseQA dataset.

        Returns:
            CommonsenseQA dataset
        """
        # Create entries
        entries = []
        for i in range(5):
            entry = DatasetEntry(
                query=f"CommonsenseQA question {i}?",
                choices={
                    "A": f"Answer A for question {i}",
                    "B": f"Answer B for question {i}",
                    "C": f"Answer C for question {i}",
                    "D": f"Answer D for question {i}",
                    "E": f"Answer E for question {i}",
                },
                metadata={"id": f"commonsense_qa_{i}", "correct_answer": "C"},
            )
            entries.append(entry)

        # Create info
        info = DatasetInfo(
            name="commonsense_qa",
            description="CommonsenseQA dataset",
            source="commonsense_qa",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        return Dataset(entries, info)


# Create the registry
REGISTRY = DatasetRegistry()


def datasets(name: str) -> Dataset:
    """Load a dataset by name.

    Args:
        name: Name of the dataset

    Returns:
        The dataset

    Raises:
        ValueError: If the dataset is not found
    """
    dataset = REGISTRY.get(name)
    if dataset is None:
        available = REGISTRY.list_datasets()
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
    return dataset


def list_available_datasets() -> List[str]:
    """List all available datasets.

    Returns:
        List of dataset names
    """
    return REGISTRY.list_datasets()


class DatasetBuilder:
    """Builder for configuring dataset loading.

    Provides a fluent interface for setting dataset loading parameters.
    """

    def __init__(self):
        """Initialize a new DatasetBuilder with default configuration."""
        self._split = None
        self._sample_size = None
        self._seed = None
        self._transform_fn = None
        self._subset = None

    def from_registry(self, dataset_name: str) -> "DatasetBuilder":
        """Select a dataset from the registry.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Self for method chaining
        """
        self._dataset_name = dataset_name
        return self

    def subset(self, subset_name: str) -> "DatasetBuilder":
        """Select a specific subset of the dataset.

        Args:
            subset_name: Name of the subset

        Returns:
            Self for method chaining
        """
        self._subset = subset_name
        return self

    def split(self, split_name: str) -> "DatasetBuilder":
        """Set the dataset split to load.

        Args:
            split_name: Name of the split (e.g., "train", "test", "validation")

        Returns:
            Self for method chaining
        """
        self._split = split_name
        return self

    def sample(self, count: int) -> "DatasetBuilder":
        """Set the number of samples to load.

        Args:
            count: Number of samples to load

        Returns:
            Self for method chaining
        """
        self._sample_size = count
        return self

    def seed(self, seed_value: int) -> "DatasetBuilder":
        """Set the random seed for reproducible sampling.

        Args:
            seed_value: Random seed value

        Returns:
            Self for method chaining
        """
        self._seed = seed_value
        random.seed(seed_value)
        return self

    def transform(
        self, transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "DatasetBuilder":
        """Apply a transformation to each dataset entry.

        Args:
            transform_fn: Function to transform entries

        Returns:
            Self for method chaining
        """
        self._transform_fn = transform_fn
        return self

    def build(self, dataset_name: Optional[str] = None) -> Dataset:
        """Build and load the dataset with the configured parameters.

        Args:
            dataset_name: Optional name of the dataset (overrides from_registry)

        Returns:
            Loaded dataset

        Raises:
            ValueError: If the dataset is not found or loading fails
        """
        if dataset_name:
            self._dataset_name = dataset_name

        if not hasattr(self, "_dataset_name"):
            raise ValueError(
                "Dataset name must be provided via from_registry() or build()"
            )

        # Load the dataset
        dataset = datasets(self._dataset_name)

        # Apply subset filtering if specified
        if self._subset:
            filtered_entries = []
            for entry in dataset.entries:
                if entry.metadata.get("subject") == self._subset:
                    filtered_entries.append(entry)

            # Create new dataset with filtered entries
            dataset = Dataset(entries=filtered_entries, info=dataset.info)

        # Apply transformations if specified
        if self._transform_fn and callable(self._transform_fn):
            transformed_entries = []
            for entry in dataset.entries:
                # Convert to dict, apply transform, then back to DatasetEntry
                entry_dict = entry.model_dump()
                transformed = self._transform_fn(entry_dict)
                transformed_entries.append(DatasetEntry(**transformed))

            # Create new dataset with transformed entries
            dataset = Dataset(entries=transformed_entries, info=dataset.info)

        # Apply sampling if specified
        if self._sample_size is not None:
            count = min(self._sample_size, len(dataset.entries))
            sampled_entries = random.sample(dataset.entries, count)

            # Create new dataset with sampled entries
            dataset = Dataset(entries=sampled_entries, info=dataset.info)

        return dataset


def main():
    """Demonstrate the DatasetBuilder pattern from the README."""
    print("Ember Data API Example from README")
    print("=================================")

    # List available datasets
    print(f"Available datasets: {list_available_datasets()}")

    # Load a dataset using the builder pattern
    print("\n1. Loading a dataset with the builder pattern:")
    try:
        dataset = (
            DatasetBuilder()
            .from_registry("mmlu")  # Use a registered dataset
            .subset("physics")  # Select a specific subset
            .split("test")  # Choose the test split
            .sample(5)  # Random sample of 5 items
            .transform(  # Apply transformations
                lambda x: {
                    "query": f"Question: {x['query']}",
                    "choices": x["choices"],
                    "metadata": x["metadata"],
                }
            )
            .build()
        )

        print(f"Successfully loaded dataset with {len(dataset)} entries")
        print(f"Dataset info: {dataset.info}")

        # Display the entries
        for i, entry in enumerate(dataset, 1):
            print(f"\nEntry {i}:")
            print(f"Query: {entry.query}")
            print(f"Choices: {entry.choices}")
            correct_letter = entry.metadata.get("correct_answer", "")
            if correct_letter in entry.choices:
                print(f"Correct answer: {entry.choices[correct_letter]}")

    except ValueError as e:
        print(f"Error: {e}")

    # Try another dataset
    print("\n2. Using a different dataset:")
    try:
        dataset = DatasetBuilder().build("truthful_qa")
        print(f"Successfully loaded TruthfulQA dataset with {len(dataset)} entries")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
