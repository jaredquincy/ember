"""Test our fixes for the circular imports issue.

This script tests the fixes we made for the EmberSettings and Field import issues.
"""


def test_field_import():
    """Test importing Field from ember_model."""
    try:
        from ember.core.types.ember_model import EmberModel, Field

        print("✅ Successfully imported Field from ember_model")
        return True
    except Exception as e:
        print(f"❌ Failed to import Field: {e}")
        return False


def test_embersettings_import():
    """Test importing EmberSettings from proper location."""
    try:
        from ember.core.config.schema import EmberSettings

        print("✅ Successfully imported EmberSettings from core.config.schema")

        # Try creating an instance
        settings = EmberSettings()
        print(f"✅ Successfully created EmberSettings instance")
        return True
    except Exception as e:
        print(f"❌ Failed to import EmberSettings: {e}")
        return False


def test_data_api():
    """Test importing data API components."""
    try:
        from ember.core.utils.data.base.models import (
            DatasetEntry,
            DatasetInfo,
            TaskType,
        )

        print("✅ Successfully imported data base models")

        # Create simple dataset entry
        entry = DatasetEntry(
            query="What is the capital of France?",
            choices={"A": "Paris", "B": "London", "C": "Berlin", "D": "Rome"},
            metadata={"correct_answer": "A"},
        )
        print(f"✅ Successfully created DatasetEntry: {entry.query}")
        return True
    except Exception as e:
        print(f"❌ Failed to import data models: {e}")
        return False


if __name__ == "__main__":
    print("Testing fixes for circular imports...")

    success_field = test_field_import()
    success_settings = test_embersettings_import()
    success_data = test_data_api()

    if success_field and success_settings and success_data:
        print("\n✅ All tests passed! Circular import issues have been resolved.")
    else:
        print("\n❌ Some tests failed. Issues remain with circular imports.")
