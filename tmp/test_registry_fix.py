"""Test that the dataset registry is properly initialized."""


def main():
    """Test dataset registry initialization."""
    from ember.api.data import list_available_datasets, datasets

    print("Testing dataset registry initialization...")

    # Check if datasets are registered
    available = list_available_datasets()
    print(f"Available datasets: {available}")

    # Try loading one of the datasets
    if available:
        dataset_name = available[0]
        try:
            print(f"\nLoading dataset '{dataset_name}'...")
            dataset = datasets(dataset_name)
            print(f"Successfully loaded dataset with {len(dataset)} entries")
            print(f"Dataset info: {dataset.info}")

            # Show a sample entry
            if len(dataset) > 0:
                print("\nSample entry:")
                entry = dataset[0]
                print(f"Query: {entry.query}")
                print(f"Choices: {entry.choices}")
                print(f"Metadata: {entry.metadata}")

            print("\nRegistry initialization is working correctly!")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("No datasets available in registry. Initialization failed.")


if __name__ == "__main__":
    main()
