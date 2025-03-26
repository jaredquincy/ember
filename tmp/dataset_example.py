"""Example of using the simplified data API from README.md."""

from ember.api.data import DatasetBuilder


def main():
    """Run the simplified data API example."""
    # Load a dataset with the builder pattern
    try:
        dataset = (
            DatasetBuilder()
            .split("test")  # Choose the test split
            .sample(5)  # Random sample of 5 items
            .seed(42)  # Set a seed for reproducibility
            .build("mmlu")
        )  # Load the MMLU dataset

        print(f"Successfully loaded {len(dataset)} entries")

        # Show some sample data
        for i, entry in enumerate(dataset, 1):
            print(f"\nEntry {i}:")
            print(f"Query: {entry.query}")
            print(f"Choices: {entry.choices}")
            print(f"Metadata: {entry.metadata}")
            if i >= 2:  # Only show first 2 entries
                break
    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()
