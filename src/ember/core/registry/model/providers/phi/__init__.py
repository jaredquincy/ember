"""Phi model provider for local inference using transformers."""

from .phi_provider import PhiProvider
from .phi_discovery import PhiDiscoveryProvider

__all__ = ["PhiProvider", "PhiDiscoveryProvider"] 