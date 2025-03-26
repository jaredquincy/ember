"""Test the simplified data API after our fixes.

This script tests the DatasetBuilder pattern from the README.
"""

try:
    # Import directly from the API
    from ember.api.data import DatasetBuilder

    def main():
        """Test the DatasetBuilder pattern."""
        print("Testing the DatasetBuilder pattern...")

        try:
            # Use the builder pattern as shown in the README
            dataset = (
                DatasetBuilder()
                .split("test")  # Choose the test split
                .sample(5)  # Random sample of 5 items
                .seed(42)  # Set a seed for reproducibility
                .build("mmlu")
            )  # Load the MMLU dataset

            print(f"Successfully created dataset: {dataset}")
            return True
        except Exception as e:
            print(f"Error using DatasetBuilder: {e}")
            import traceback

            traceback.print_exc()
            return False

    if __name__ == "__main__":
        success = main()
        if success:
            print("\n✅ Successfully used the DatasetBuilder pattern!")
        else:
            print("\n❌ Failed to use the DatasetBuilder pattern.")
except ImportError as e:
    print(f"Failed to import from ember.api.data: {e}")
    print(
        "This may be due to remaining circular import issues that weren't fixed by our changes."
    )
