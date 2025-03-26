# Remaining Issues in Ember's Data API

While we've fixed the immediate circular import issues with EmberSettings and Field, there are some remaining issues that would need to be addressed for a fully functional data API:

## 1. ModelDiscoveryError Import Issue

```
Error during model discovery: cannot import name 'ModelDiscoveryError' from 'ember.core.registry.model.providers.base_discovery'
```

This suggests there's another circular import or missing definition involving `ModelDiscoveryError`. This error would need to be investigated and fixed similar to how we fixed the EmberSettings issue.

## 2. Dataset Registry Integration

The data API depends on a properly initialized dataset registry. When attempting to use the DatasetBuilder, we got:

```
Dataset 'mmlu' not found. Available datasets: []
```

This indicates that the dataset registry isn't properly initialized or populated. This would require:

1. Ensuring datasets are registered correctly
2. Setting up registry initialization during module import
3. Potentially handling the case when datasets aren't found with better error messages

## 3. Integration with Full Ember Framework

Our tests showed that the basic components (`DatasetEntry`, `DatasetInfo`, etc.) work fine on their own, but full integration with the Ember framework (model registry, etc.) might still have issues.

## 4. Cleaner Import Structure

The current import structure is prone to circular imports. A clearer layering would be beneficial:

1. **Base Layer**: Core types, protocols, exceptions (no dependencies)
2. **Config Layer**: Configuration schemas and settings (depends only on Base)
3. **Core Layer**: Core functionality like data models, transformers (depends on Base and Config)
4. **API Layer**: Public API facades (depends on Core)

## 5. Model Discovery Issues

There appears to be an issue with model discovery during initialization. This affects the overall framework but is tangential to the data API issues.

## 6. Documentation Updates

The API documentation and examples need updating to reflect the correct import patterns and usage.

## 7. Standalone Usage Patterns

Consider providing simplified patterns for standalone usage of components like the data API without needing the full framework initialized.

## Next Steps

1. Address the `ModelDiscoveryError` import issue
2. Set up proper dataset registration for testing
3. Continue refactoring the import structure for cleaner dependency management
4. Add more comprehensive tests that verify proper integration
5. Update documentation to reflect the correct usage patterns