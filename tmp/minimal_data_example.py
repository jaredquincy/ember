"""Minimal data example using core utilities directly."""

from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from ember.core.utils.data.service import DatasetService
from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader
from ember.core.utils.data.base.samplers import DatasetSampler
from ember.core.utils.data.base.validators import DatasetValidator
from ember.core.utils.data.base.preppers import DatasetPrepper


def create_mock_dataset():
    """Create a mock dataset for demonstration."""
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": 2,  # Paris
            "category": "Geography",
        },
        {
            "question": "Which of these is a mammal?",
            "choices": ["Shark", "Snake", "Eagle", "Dolphin"],
            "answer": 3,  # Dolphin
            "category": "Biology",
        },
        {
            "question": "What is the square root of 144?",
            "choices": ["10", "12", "14", "16"],
            "answer": 1,  # 12
            "category": "Mathematics",
        },
    ]


class SimplePrepper(DatasetPrepper):
    """A simple dataset prepper for the mock data."""

    def get_required_keys(self):
        """Return the keys required for this dataset."""
        return ["question", "choices", "answer"]

    def create_dataset_entries(self, item):
        """Create dataset entries from a data item."""
        choices_dict = {chr(65 + i): choice for i, choice in enumerate(item["choices"])}

        correct_letter = chr(65 + item["answer"])

        entry = DatasetEntry(
            query=item["question"],
            choices=choices_dict,
            metadata={
                "correct_answer": correct_letter,
                "category": item.get("category", "Unknown"),
            },
        )
        return [entry]


def main():
    """Run the minimal data example."""
    try:
        # Create mock dataset
        mock_data = create_mock_dataset()

        # Set up dataset info
        dataset_info = DatasetInfo(
            name="mock_test",
            description="A mock test dataset",
            source="mock_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Create service components
        prepper = SimplePrepper()
        loader = HuggingFaceDatasetLoader()  # Won't actually be used with our mock data
        validator = DatasetValidator()
        sampler = DatasetSampler()

        # Initialize service
        service = DatasetService(loader=loader, validator=validator, sampler=sampler)

        # Process mock data (bypassing the actual loader)
        entries = service._prep_data(
            dataset_info=dataset_info, sampled_data=mock_data, prepper=prepper
        )

        # Display results
        print(f"Processed {len(entries)} dataset entries:\n")
        for i, entry in enumerate(entries, 1):
            print(f"Entry {i}:")
            print(f"  Query: {entry.query}")
            print(f"  Choices: {entry.choices}")
            print(f"  Correct answer: {entry.metadata['correct_answer']}")
            print(f"  Category: {entry.metadata['category']}")
            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
