# Ember Examples Debugging

This document tracks the troubleshooting process for fixing failing examples in the Ember framework.

## 1. Operator Composition Example (`composition_example.py`)

### Issue
The example fails with an error when attempting to invoke the "openai:gpt-4o" model.

### Root Cause Analysis
1. **Incorrect Parameter Name**: The `LMModuleConfig` class expects an `id` field, but the code uses `model_name`:
   ```python
   self.lm_module = LMModule(
       config=LMModuleConfig(
           model_name=model_name,  # INCORRECT: should be "id" not "model_name"
           temperature=temperature,
       )
   )
   ```

2. **No Error Handling**: The `QuestionRefinement` operator doesn't include any error handling for model invocation, making the example fragile.

### Fix Plan
1. Update parameter name in `LMModuleConfig` from `model_name` to `id`
2. Add proper error handling to the `QuestionRefinement.forward` method
3. Update the model name in `main()` to ensure it's available (e.g., use "openai:gpt-3.5-turbo" as it's more likely to be available)

## 2. Transformation Example (`transformation_example.py`)

### Issue
The example fails with an error related to invalid input/output types in the `SimpleOperator` class within the `demonstrate_vmap` function.

### Root Cause Analysis
1. **Type Mismatch in vmap**: The `vmap` transformation returns a dictionary of results, but the operator's specification expects a `SimpleOutput` model.

2. **Missing Type Conversion**: The `_combine_outputs` function in `vmap.py` doesn't convert the combined dictionary results back to the expected model type.

### Fix Plan
1. Modify the `_combine_outputs` function in `vmap.py` to check the operator's `specification.structured_output` and convert the combined result dictionary to the appropriate model type.
2. Alternatively, update the `SimpleOperator` class to correctly handle the dictionary output from vmap.

## Architectural Considerations

1. **Unified Error Handling**: Consider a standardized approach for error handling in operators that use LLMs.
2. **Type Compatibility**: Ensure transformations like `vmap` properly respect the type specifications of operators.
3. **Graceful Degradation**: Examples should have graceful error handling to make them more robust, especially for cases involving external APIs.
4. **Automated Testing**: Add tests to catch these issues before they appear in examples.