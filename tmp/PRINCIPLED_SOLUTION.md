# Principled Solution for Ember Data API

## Root Cause Analysis

The current architecture has several systemic issues:

1. **Ambiguous Initialization**: Multiple initialization paths with unclear precedence
2. **Documentation-Implementation Mismatch**: README describes API features that don't exist
3. **Circular Dependencies**: Core modules import from each other creating subtle circular dependencies
4. **Registry Exposure**: Global registry singleton accessed directly from multiple places

## Architectural Improvements

### 1. Single Entry Point Pattern

```python
# In ember/api/data.py

# Dedicated initialization function with clear ownership
def _ensure_initialized():
    """Ensure dataset registry is initialized exactly once."""
    global _initialized
    if not _initialized:
        from ember.core.utils.data.registry import initialize_registry
        initialize_registry()
        _initialized = True

# Singleton pattern with lazy initialization
_initialized = False
```

### 2. Clear Registry Access Pattern

```python
# In ember/api/data.py

def datasets(name: str, config = None) -> Dataset:
    """Load a dataset by name with optional configuration.
    
    Ensures registry is initialized first, then delegates to core implementation.
    """
    _ensure_initialized()  # Guarantees initialization happens before access
    
    # Implementation follows...
```

### 3. Implementing the Builder Pattern Properly

```python
# In ember/api/data.py

class DatasetBuilder:
    """Builder for dataset configuration.
    
    Implementation exactly matches the documented API in README.md
    """
    
    def __init__(self):
        # Initialize with null state
        self._dataset_name = None
        # Other fields...
    
    def from_registry(self, dataset_name: str) -> "DatasetBuilder":
        # Explicitly documented method
        self._dataset_name = dataset_name
        return self
        
    def subset(self, subset_name: str) -> "DatasetBuilder":
        # Explicitly documented method
        self._subset = subset_name
        return self
    
    # Other methods...
```

## Implementation Plan

1. **Fix Initialization**: Add lazy initialization to public API
2. **Update Builder API**: Ensure implementation matches documentation
3. **Simplify Registry Access**: Centralize access through public API
4. **Clear Documentation**: Update code comments to reflect the design intent
5. **Add Tests**: Add targeted tests to verify initialization behavior

## Improvements Over Current Approach

1. **Predictable Initialization**: Happens exactly once, at the right time
2. **Simplified Dependency Flow**: Core → API, never API → Core → API
3. **Decoupled Implementation**: Registry is an implementation detail, not exposed
4. **Matches Documentation**: Implementation matches the behavior shown in README
5. **Maintainable Code**: Clear ownership of initialization logic

## Proposed Changes

### 1. Update ember/api/data.py

```python
"""Data API for Ember.

This module provides a simplified interface to Ember's data utilities.
It ensures the dataset registry is properly initialized and exposes a
clean API that matches the documentation.
"""

# Core imports with clear dependency direction
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, Callable

from ember.core.utils.data.base.config import BaseDatasetConfig as DatasetConfig
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo, TaskType
from ember.core.utils.data.registry import UNIFIED_REGISTRY, register

# Initialization state
_initialized = False

# Initialization function with clear ownership
def _ensure_initialized():
    """Ensure the dataset registry is initialized exactly once."""
    global _initialized
    if not _initialized:
        from ember.core.utils.data.registry import initialize_registry
        initialize_registry()
        _initialized = True

# The rest of the implementation follows...
```

This principled approach creates a clear architectural foundation with proper initialization, matching the documented API, and following software engineering best practices.