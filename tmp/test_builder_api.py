"""Test the DatasetBuilder API from the README."""

from ember.api.data import DatasetBuilder


def main():
    """Test the DatasetBuilder API from the README."""
    print("Testing DatasetBuilder API from README...")

    try:
        # Create the builder pattern from the README
        dataset = (
            DatasetBuilder()
            .from_registry("mmlu")  # Use a registered dataset
            .subset("physics")  # Select a specific subset
            .split("test")  # Choose the test split
            .sample(100)  # Random sample of 100 items
            .transform(  # Apply transformations
                lambda x: {
                    "query": f"Question: {x['query']}",
                    "choices": x["choices"],
                    "metadata": x["metadata"],
                }
            )
            .build()
        )

        print(f"Successfully created dataset with {len(dataset)} entries")
        print(f"Dataset info: {dataset.info}")

        # Show sample entries
        if len(dataset) > 0:
            print("\nSample entry:")
            entry = dataset[0]
            print(f"Query: {entry.query}")
            print(f"Choices: {entry.choices}")
            print(f"Metadata: {entry.metadata}")

        return True
    except Exception as e:
        print(f"Error using DatasetBuilder: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ DatasetBuilder API works correctly!")
    else:
        print("\n❌ Issues with DatasetBuilder API.")
