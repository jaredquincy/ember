"""
Logging Configuration Module for Ember

This module centralizes logging configuration for the Ember framework,
providing consistent logging behavior across all components while allowing
for appropriate verbosity control.

The design follows standard Python logging best practices:
- Libraries (like Ember) should configure loggers but not handlers
- Applications control how logs are displayed/stored
- Sensible defaults are provided but easily overridable

Usage:
    # To apply standard configuration with reduced verbosity:
    from ember.core.utils.logging import configure_logging
    configure_logging(verbose=False)

    # To adjust specific component verbosity:
    from ember.core.utils.logging import set_component_level
    set_component_level("model_discovery", logging.DEBUG)
"""

import logging

# Component groups allow configuring related loggers together
COMPONENT_GROUPS = {
    "model_registry": [
        "ember.core.registry.model",
        "ember.core.registry.model.initialization",
        "ember.core.registry.model.base.registry.discovery",
        "ember.core.registry.model.base.registry.model_registry",
    ],
    "model_warnings": [
        # These loggers produce too many warnings during discovery
        "ember.core.registry.model.base.registry.discovery",
    ],
    "model_discovery": [
        "ember.core.registry.model.providers.anthropic.anthropic_discovery",
        "ember.core.registry.model.providers.openai.openai_discovery",
        "ember.core.registry.model.providers.deepmind.deepmind_discovery",
    ],
    "http": [
        "httpcore",
        "httpx",
        "urllib3",
        "openai",
    ],
}

# Reverse mapping to find group by logger name
LOGGER_TO_GROUP = {}
for group_name, loggers in COMPONENT_GROUPS.items():
    for logger_name in loggers:
        LOGGER_TO_GROUP[logger_name] = group_name


def configure_logging(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    verbose: bool = False,
) -> None:
    """
    Configure logging for the Ember framework.

    This function sets appropriate logging levels for all Ember components
    and external dependencies. It does not configure handlers (following Python
    best practices for libraries).

    Args:
        level: Base logging level for all components (default: INFO).
        format_str: Log format string (only used for root logger).
        verbose: If False, reduces verbosity for non-essential components.
    """
    # Set base logging level for Ember
    ember_logger = logging.getLogger("ember")
    ember_logger.setLevel(level)

    # Configure root logger format only if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(handler)

    # Set component-specific levels based on verbosity
    quiet_level = logging.INFO if verbose else logging.WARNING

    # Model registry and discovery are typically verbose and not needed at INFO level
    set_component_group_level("model_registry", quiet_level)
    set_component_group_level("model_discovery", quiet_level)

    # HTTP libraries are very verbose at INFO level
    set_component_group_level("http", quiet_level)
    
    # Add NullHandler to HTTP libraries to prevent "no handler" warnings
    # and shutdown logging errors
    _configure_http_library_handlers()

    # Some components need even stricter logging in non-verbose mode
    if not verbose:
        # Discovery warnings are too numerous during model initialization
        set_component_group_level("model_warnings", logging.ERROR)


def set_component_level(component: str, level: int) -> None:
    """
    Set logging level for a specific Ember component.

    Args:
        component: Component name or logger name
        level: Logging level to set (e.g., logging.INFO, logging.WARNING)
    """
    # If component is a known group, set the entire group
    if component in COMPONENT_GROUPS:
        set_component_group_level(component, level)
        return

    # Otherwise set individual logger
    logger = logging.getLogger(component)
    logger.setLevel(level)


def set_component_group_level(group_name: str, level: int) -> None:
    """
    Set logging level for an entire component group.

    Args:
        group_name: Name of component group
        level: Logging level to set
    """
    if group_name not in COMPONENT_GROUPS:
        raise ValueError(f"Unknown component group: {group_name}")

    for logger_name in COMPONENT_GROUPS[group_name]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def _configure_http_library_handlers() -> None:
    """
    Prevent HTTP library logging errors during shutdown.
    
    Root cause: The issue stems from httpcore's trace system logging debug messages during
    garbage collection at interpreter shutdown, when output streams may already be closed.
    
    This solution directly patches Python's StreamHandler.emit method to gracefully handle
    closed streams during shutdown - the most precise point to fix the issue.
    """
    # First, silence the specific loggers responsible for most shutdown errors
    for name in ["httpcore.connection", "httpcore.http11"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # Only patch once to avoid re-patching
    if not hasattr(logging, "_ember_patched_handlers"):
        # Store original method
        original_emit = logging.StreamHandler.emit
        
        # Create resilient version of emit that handles closed streams
        def safe_emit(self, record):
            """StreamHandler.emit that gracefully handles closed streams during shutdown."""
            try:
                original_emit(self, record)
            except (ValueError, IOError, OSError) as e:
                # Only suppress errors related to closed files during shutdown
                if not (isinstance(e, ValueError) and "closed file" in str(e).lower()):
                    if not (isinstance(e, (IOError, OSError)) and "closed" in str(e).lower()):
                        raise
        
        # Apply patch
        logging.StreamHandler.emit = safe_emit
        
        # Mark as patched so we don't patch twice
        logging._ember_patched_handlers = True


def get_ember_logger(name: str) -> logging.Logger:
    """
    Get a logger with the ember namespace.

    Args:
        name: Logger name (will be prefixed with 'ember.' if not already)

    Returns:
        Configured logger instance
    """
    if not name.startswith("ember."):
        name = f"ember.{name}"
    return logging.getLogger(name)
