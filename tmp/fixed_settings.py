"""Temporary fix for the EmberSettings import issue.

This creates a simple version of EmberSettings that can be imported to fix the circular import.
We'll use this as a test to confirm that this resolves the import error.
"""

from ember.core.config.schema import EmberConfig


class EmberSettings(EmberConfig):
    """Configuration settings for Ember.

    This class extends EmberConfig to provide a more user-friendly API for configuration.
    It resolves a circular import issue with EmberSettings referenced in model configuration.

    Usage:
        settings = EmberSettings()
        settings.registry.auto_discover = True
    """

    pass
