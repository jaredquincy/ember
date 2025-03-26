"""Test the EmberSettings fix.

This script tests importing the fixed EmberSettings class to resolve the circular import.
"""

import sys
import os

# Add the tmp directory to path so we can import our fixed module
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import our fixed EmberSettings
from fixed_settings import EmberSettings


# Now import the module that was failing
def test_import():
    """Test importing the problematic module."""
    try:
        from ember.core.registry.model.base.registry.discovery import (
            ModelDiscoveryService,
        )

        print("✅ Successfully imported ModelDiscoveryService")

        # Try creating an instance to verify it works
        service = ModelDiscoveryService()
        print("✅ Successfully created ModelDiscoveryService instance")

        # Import the module that was referencing EmberSettings
        from ember.core.registry.model.__init__ import ModelRegistry

        print("✅ Successfully imported ModelRegistry")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check if we can import the fixed module
    print("Testing EmberSettings fix...")
    print(f"Using fixed_settings.py from {os.path.abspath('fixed_settings.py')}")

    # Attempt to create an EmberSettings instance
    settings = EmberSettings()
    print(f"✅ Successfully created EmberSettings instance: {settings}")

    # Test importing the problematic module
    success = test_import()
    print(f"Import test {'succeeded' if success else 'failed'}")
