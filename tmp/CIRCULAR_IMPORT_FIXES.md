# Circular Import Fixes in Ember

## Summary of Issues Fixed

1. **EmberSettings Circular Import**:
   - Issue: `EmberSettings` was referenced in `discovery.py` but was missing in `settings.py`
   - Fix: Moved `EmberSettings` class to `ember.core.config.schema` where it logically belongs
   - Updated all references to import from the new location

2. **Field Import Missing**:
   - Issue: `Field` was imported from `ember.core.types.ember_model` but wasn't defined there
   - Fix: Added the Pydantic `Field` import to the `ember_model.py` file

## Changes Made

1. **Added Field import in ember_model.py**:
   ```python
   from pydantic import BaseModel, ConfigDict, Field
   ```

2. **Added EmberSettings to ember.core.config.schema**:
   ```python
   class EmberSettings(EmberConfig):
       """Configuration settings for Ember.
   
       This class extends EmberConfig to provide a more user-friendly API for configuration.
       It resolves circular import issues with model configuration.
   
       Usage:
           settings = EmberSettings()
           settings.registry.auto_discover = True
       """
       pass
   ```

3. **Updated import in discovery.py**:
   ```python
   from ember.core.config.schema import EmberSettings
   ```

4. **Updated import in model/__init__.py**:
   ```python
   # Configuration and initialization - import from core config to avoid circular imports
   from ember.core.config.schema import EmberSettings
   ```

5. **Removed temporary EmberSettings from settings.py**:
   Added a comment explaining the class is now defined elsewhere.

## Testing

Successfully verified the fixes with:

1. **Basic Component Tests**:
   - Successfully imported `Field` from `ember_model`
   - Successfully imported `EmberSettings` from `core.config.schema`
   - Successfully imported data base models

2. **API Usage Test**:
   - DatasetBuilder API works now - though we receive an expected error about "Dataset 'mmlu' not found" which is unrelated to the import issues and indicates the API is working correctly

## Root Cause Analysis

The circular imports were caused by:

1. **Improper Layering**: 
   - Core components (like config schemas) were importing from higher-level modules
   - Discovery code was looking for `EmberSettings` in the wrong location

2. **Missing Re-exports**:
   - The `Field` class from Pydantic should have been re-exported in `ember_model.py`

3. **Dependency Structure**:
   - The core issue was that classes and imports were not properly organized according to dependency direction
   - Settings classes should be in a foundational layer that other modules can import from

## Recommendations for Preventing Future Issues

1. **Clearer Layering**:
   - Define clear dependency layers in the codebase
   - Use absolute imports consistently
   - Avoid circular dependencies by proper structuring

2. **Dependency Direction**:
   - Base configuration and settings should be foundational components
   - Higher-level modules should depend on lower-level ones, not vice versa

3. **Import Testing**:
   - Add specific tests for imports to catch circular dependencies
   - Use tools like `import-linter` to enforce proper dependency direction

4. **Documentation**:
   - Clearly document the expected import patterns
   - Update module docstrings to show exactly what to import and from where