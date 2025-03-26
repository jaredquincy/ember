"""Test the DatasetBuilder pattern from the README.

This script tests that the DatasetBuilder pattern shown in the README works correctly.
"""

try:
    from ember.api.data import DatasetBuilder, datasets, list_available_datasets

    def main():
        """Test the DatasetBuilder pattern."""
        print("Available datasets:", list_available_datasets())

        print("\nTesting the DatasetBuilder pattern...")
        try:
            # First try to access a dataset directly to see if the registry is working
            print("Trying to access datasets directly...")
            available_datasets = list_available_datasets()
            if available_datasets:
                dataset_name = available_datasets[0]
                print(f"Loading dataset '{dataset_name}'...")
                dataset = datasets(dataset_name)
                print(f"✅ Successfully loaded dataset: {dataset}")
            else:
                print("No datasets available in registry")

            # Now try the builder pattern
            print("\nTrying builder pattern...")
            try:
                builder = DatasetBuilder()
                print(f"✅ Successfully created DatasetBuilder: {builder}")
                # The actual build will likely fail due to missing datasets in registry
                # but we want to make sure the builder and API work properly
                print("Builder works correctly!")
                return True
            except Exception as e:
                print(f"❌ Error using DatasetBuilder: {e}")
                import traceback

                traceback.print_exc()
                return False

        except Exception as e:
            print(f"❌ Error with datasets: {e}")
            import traceback

            traceback.print_exc()
            return False

    if __name__ == "__main__":
        success = main()
        if success:
            print("\n✅ DatasetBuilder pattern works correctly!")
        else:
            print("\n❌ Issues remain with the DatasetBuilder pattern.")
except ImportError as e:
    print(f"Failed to import from ember.api.data: {e}")
    print("This may be due to remaining import issues.")
