"""Test our fixes for the ModelDiscoveryError import issue.

This script tests that we've resolved the ModelDiscoveryError import issue.
"""


def test_model_discovery_imports():
    """Test importing model discovery classes."""
    try:
        from ember.core.registry.model.providers.base_discovery import (
            BaseDiscoveryProvider,
        )

        print("✅ Successfully imported BaseDiscoveryProvider")

        # Try importing a specific provider
        from ember.core.registry.model.providers.anthropic.anthropic_discovery import (
            AnthropicDiscovery,
        )

        print("✅ Successfully imported AnthropicDiscovery")

        # Import the error directly
        from ember.core.exceptions import ModelDiscoveryError

        print("✅ Successfully imported ModelDiscoveryError")

        return True
    except Exception as e:
        print(f"❌ Failed to import model discovery classes: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing fixes for ModelDiscoveryError issue...")

    success = test_model_discovery_imports()

    if success:
        print(
            "\n✅ All imports successful! The ModelDiscoveryError issue has been resolved."
        )
    else:
        print("\n❌ Import tests failed. Issues remain with ModelDiscoveryError.")
