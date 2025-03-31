# Dataset Implementation Plan: AIME 2024, GPQA Diamond, and Codeforces

This document outlines the implementation plan for integrating three new datasets into the Ember framework: AIME 2024 (Competition Math), GPQA Diamond (PhD-Level Science Questions), and Codeforces (Competitive Programming Problems).

## Table of Contents

1. [Overview](#1-overview)
2. [Implementation Structure](#2-implementation-structure)
3. [Detailed Implementation Plan](#3-detailed-implementation-plan)
   - [3.1. AIME 2024](#31-aime-2024-competition-math)
   - [3.2. GPQA Diamond](#32-gpqa-diamond-phd-level-science-questions)
   - [3.3. Codeforces](#33-codeforces-competitive-programming-problems)
4. [Implementation Steps](#4-implementation-steps)
5. [Testing Strategy](#5-testing-strategy)
6. [Implementation Details](#6-implementation-details)
   - [6.1. AIME 2024 Implementation](#61-aime-2024-implementation)
   - [6.2. GPQA Diamond Implementation](#62-gpqa-diamond-implementation)
   - [6.3. Codeforces Implementation](#63-codeforces-implementation)
   - [6.4. Custom Evaluator for Codeforces](#64-custom-evaluator-for-codeforces)
7. [Integration Examples](#7-integration-examples)
8. [Documentation](#8-documentation)
9. [Code Style and Engineering Quality](#9-code-style-and-engineering-quality)
10. [Timeline and Milestones](#10-timeline-and-milestones)
11. [Future Extensions](#11-future-extensions)
12. [Implementation Progress](#12-implementation-progress)

## 1. Overview

This plan details the integration of three datasets into Ember, following the architectural patterns and coding standards already established in the codebase. Each dataset will have its own dedicated implementation in the datasets registry, with appropriate preppers, configuration, and evaluation mechanisms.

## 2. Implementation Structure

For each dataset, we'll create:

1. **Dataset Prepper**: A class that transforms raw dataset entries into Ember's standardized `DatasetEntry` format
2. **Dataset Configuration** (if needed): For datasets with configurable options
3. **Dataset Registration**: Integration with Ember's registration system
4. **Evaluation Strategy**: Implementation of appropriate evaluators for each dataset type
5. **Documentation**: Comprehensive docstrings and comments following the Jeff Dean/Sanjay Ghemawat style

## 3. Detailed Implementation Plan

### 3.1. AIME 2024 (Competition Math)

#### Implementation Files
1. Create `src/ember/core/utils/data/datasets_registry/aime.py`

#### AIME Dataset Structure
```python
class AIMEConfig(BaseDatasetConfig):
    """Configuration for the AIME dataset.
    
    Allows loading either all problems or specific years.
    """
    year: Optional[int] = None  # Specific year filter (2024 by default)
    contest: Optional[str] = None  # 'I' or 'II' for specific contest

class AIMEPrepper(IDatasetPrepper):
    """Prepares AIME competition math problems.
    
    Transforms HuggingFace AIME dataset entries into DatasetEntry format.
    """
    # Implementation will handle processing LaTeX-formatted problems

# Registration
register_metadata(
    name="aime",
    description="American Invitational Mathematics Examination questions",
    source="Maxwell-Jia/AIME_2024",
    task_type=TaskType.SHORT_ANSWER,
    prepper_class=AIMEPrepper,
)
```

#### Evaluation Implementation
```python
class NumericAnswerEvaluator(IEvaluator[str, str]):
    """Evaluator for numeric answers that handles exact integer matching.
    
    Extracts numeric values from model responses and compares with reference.
    """
    # Implementation will handle integer comparison for AIME answers
```

### 3.2. GPQA Diamond (PhD-Level Science Questions)

#### Implementation Files
1. Create `src/ember/core/utils/data/datasets_registry/gpqa.py`

#### GPQA Dataset Structure
```python
class GPQAConfig(BaseDatasetConfig):
    """Configuration for the GPQA dataset.
    
    Allows loading specifically the Diamond subset or other configurations.
    """
    subset: str = "gpqa_diamond"  # Default to the diamond subset
    difficulty: Optional[str] = None  # Optional filter by difficulty

class GPQAPrepper(IDatasetPrepper):
    """Prepares GPQA science questions.
    
    Transforms HuggingFace GPQA dataset entries into DatasetEntry format.
    """
    # Implementation will handle multiple-choice structure

# Registration
register_metadata(
    name="gpqa",
    description="Graduate-level PhD science questions (physics, chemistry, biology)",
    source="Idavidrein/gpqa",
    task_type=TaskType.MULTIPLE_CHOICE,
    prepper_class=GPQAPrepper,
)
```

#### Evaluation Implementation
Will use existing `ExactMatchEvaluator` for multiple-choice evaluation.

### 3.3. Codeforces (Competitive Programming Problems)

#### Implementation Files
1. Create `src/ember/core/utils/data/datasets_registry/codeforces.py`
2. Create supporting evaluation classes in `src/ember/core/utils/eval/code_execution.py`

#### Codeforces Dataset Structure
```python
class CodeForcesConfig(BaseDatasetConfig):
    """Configuration for the Codeforces dataset.
    
    Allows filtering by difficulty, tags, or other attributes.
    """
    difficulty_range: Optional[Tuple[int, int]] = None  # Range of difficulty ratings
    tags: Optional[List[str]] = None  # Problem tags for filtering
    limit: Optional[int] = None  # Max number of problems to include

class CodeForcesPrepper(IDatasetPrepper):
    """Prepares Codeforces programming problems.
    
    Transforms HuggingFace Codeforces dataset entries into DatasetEntry format.
    """
    # Implementation will handle problem statement, input specs, test cases

# Registration
register_metadata(
    name="codeforces",
    description="Competitive programming problems from Codeforces",
    source="open-r1/codeforces",
    task_type=TaskType.CODE_COMPLETION,
    prepper_class=CodeForcesPrepper,
)
```

#### Evaluation Implementation
```python
class CodeCompetitionEvaluator(IEvaluator[str, Dict[str, Any]]):
    """Evaluator for competitive programming problems.
    
    Executes student code against multiple test cases with safety controls.
    Handles input/output format validation and time/space complexity analysis.
    """
    # Implementation with sandboxed code execution and test case validation
```

## 4. Implementation Steps

1. **Setup & Structure** (1 day)
   - Create skeleton files for each dataset
   - Define interfaces and base classes
   - Create test scaffolding

2. **AIME 2024 Implementation** (1 day)
   - Implement AIMEConfig and AIMEPrepper
   - Add registration code
   - Create evaluator for math answers
   - Add tests

3. **GPQA Diamond Implementation** (1 day)
   - Implement GPQAConfig and GPQAPrepper
   - Add registration code
   - Test with existing MCQ evaluators
   - Add tests

4. **Codeforces Implementation** (2 days)
   - Implement CodeForcesConfig and CodeForcesPrepper
   - Create code execution evaluator with safety controls
   - Integrate test case validation
   - Add tests

5. **Integration & Testing** (1 day)
   - End-to-end tests for all three datasets
   - Performance optimization
   - Documentation review

6. **Example Notebooks & Documentation** (1 day)
   - Create example usage notebooks
   - Add comprehensive documentation
   - Update API references

## 5. Testing Strategy

For each dataset:

1. **Unit Tests**
   - Test prepper functionality in isolation
   - Test evaluator functionality in isolation
   - Test configuration validation

2. **Integration Tests**
   - End-to-end dataset loading tests
   - Evaluation pipeline tests

3. **Performance Tests**
   - Benchmark dataset loading speeds
   - Optimize for large datasets

## 6. Implementation Details

### 6.1. AIME 2024 Implementation

```python
from typing import Any, Dict, List, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.registry import register_metadata
from ember.core.utils.data.base.models import TaskType


class AIMEConfig(BaseDatasetConfig):
    """Configuration for the AIME dataset.
    
    Controls filtering and loading options for the AIME math competition dataset.
    """
    year: Optional[int] = 2024  # Default to 2024
    contest: Optional[str] = None  # 'I' or 'II' for specific contest
    
    
class AIMEPrepper(IDatasetPrepper):
    """Prepper for AIME competition math problems.
    
    Transforms the HuggingFace AIME dataset entries into Ember's DatasetEntry format,
    handling LaTeX-formatted math problems and numeric answers.
    """
    
    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the AIME prepper with optional configuration.
        
        Args:
            config: Either a string (year), AIMEConfig instance, or None.
                   If None, defaults to all 2024 problems.
        """
        if isinstance(config, str) and config.isdigit():
            config = AIMEConfig(year=int(config))
        elif config is None:
            config = AIMEConfig()
        super().__init__(config)
        self.year = self._config.year
        self.contest = self._config.contest
        
    def get_required_keys(self) -> List[str]:
        """Return required keys for AIME dataset items.
        
        Returns:
            List of required fields: ID, Problem, Answer
        """
        return ["ID", "Problem", "Answer"]
    
    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a DatasetEntry from an AIME problem.
        
        Args:
            item: Raw dataset item containing ID, Problem, Answer and Solution.
                 
        Returns:
            DatasetEntry with the problem as query and answer in metadata.
        """
        problem_id = str(item["ID"])
        problem_text = str(item["Problem"])
        answer = str(item["Answer"])
        solution = item.get("Solution", "")
        
        # Filter by year/contest if specified
        if self.year or self.contest:
            # AIME IDs have format "YYYY-C-N" where C is contest (I/II) and N is problem number
            parts = problem_id.split("-")
            if len(parts) >= 3:
                id_year = int(parts[0]) if parts[0].isdigit() else None
                id_contest = parts[1]
                
                if (self.year and id_year != self.year) or \
                   (self.contest and id_contest != self.contest):
                    return []  # Skip this problem
        
        return [
            DatasetEntry(
                query=problem_text,
                choices={},  # No choices for short answer problems
                metadata={
                    "correct_answer": answer,
                    "solution": solution,
                    "problem_id": problem_id,
                }
            )
        ]


# Register the AIME dataset
register_metadata(
    name="aime",
    description="American Invitational Mathematics Examination problems",
    source="Maxwell-Jia/AIME_2024",
    task_type=TaskType.SHORT_ANSWER,
    prepper_class=AIMEPrepper,
)
```

### 6.2. GPQA Diamond Implementation

```python
from typing import Any, Dict, List, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.registry import register_metadata
from ember.core.utils.data.base.models import TaskType


class GPQAConfig(BaseDatasetConfig):
    """Configuration for the GPQA dataset.
    
    Controls loading options for the GPQA PhD-level science questions.
    """
    subset: str = "gpqa_diamond"  # Default to Diamond subset
    difficulty: Optional[str] = None  # Optional filter by difficulty level


class GPQAPrepper(IDatasetPrepper):
    """Prepper for GPQA Diamond science questions.
    
    Transforms HuggingFace GPQA dataset entries into DatasetEntry format,
    handling the multiple-choice structure with correct and incorrect answers.
    """
    
    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the GPQA prepper with optional configuration.
        
        Args:
            config: Either a string (subset name), GPQAConfig instance, or None.
                   If None, defaults to Diamond subset.
        """
        if isinstance(config, str):
            config = GPQAConfig(subset=config)
        elif config is None:
            config = GPQAConfig()
        super().__init__(config)
        self.subset = self._config.subset
        self.difficulty = self._config.difficulty
        
    def get_required_keys(self) -> List[str]:
        """Return required keys for GPQA dataset items.
        
        Returns:
            List of required fields for processing.
        """
        return ["question", "correct_answer", "incorrect_answers"]
    
    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a DatasetEntry from a GPQA question.
        
        Args:
            item: Raw dataset item containing question text and answers.
                 
        Returns:
            DatasetEntry with the question as query and answer choices.
        """
        question = str(item["question"])
        correct_answer = str(item["correct_answer"])
        incorrect_answers = [str(ans) for ans in item["incorrect_answers"]]
        
        # Apply difficulty filter if specified
        if self.difficulty and item.get("difficulty") != self.difficulty:
            return []  # Skip this question
        
        # Create choices dictionary with letter keys (A, B, C, D)
        all_choices = [correct_answer] + incorrect_answers
        choices = {chr(65 + i): choice for i, choice in enumerate(all_choices)}
        
        # The correct answer is always the first one (A)
        return [
            DatasetEntry(
                query=question,
                choices=choices,
                metadata={
                    "correct_answer": "A",  # The first choice is always correct in our formatting
                    "subject": item.get("subject", ""),
                    "difficulty": item.get("difficulty", ""),
                }
            )
        ]


# Register the GPQA dataset
register_metadata(
    name="gpqa",
    description="Graduate-level PhD science questions (GPQA Diamond subset)",
    source="Idavidrein/gpqa",
    task_type=TaskType.MULTIPLE_CHOICE,
    prepper_class=GPQAPrepper,
)
```

### 6.3. Codeforces Implementation

```python
from typing import Any, Dict, List, Optional, Tuple

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.registry import register_metadata
from ember.core.utils.data.base.models import TaskType


class CodeForcesConfig(BaseDatasetConfig):
    """Configuration for the Codeforces dataset.
    
    Controls filtering and loading options for competitive programming problems.
    """
    difficulty_range: Optional[Tuple[int, int]] = None  # Min/max difficulty
    tags: Optional[List[str]] = None  # Problem tags for filtering
    limit: Optional[int] = None  # Max number of problems


class CodeForcesPrepper(IDatasetPrepper):
    """Prepper for Codeforces programming problems.
    
    Transforms HuggingFace Codeforces dataset entries into DatasetEntry format,
    handling problem statements, input specifications, and test cases.
    """
    
    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the Codeforces prepper with optional configuration.
        
        Args:
            config: CodeForcesConfig instance or None for defaults.
        """
        if config is None:
            config = CodeForcesConfig()
        super().__init__(config)
        self.difficulty_range = self._config.difficulty_range
        self.tags = self._config.tags
        self.limit = self._config.limit
        self.problem_count = 0
        
    def get_required_keys(self) -> List[str]:
        """Return required keys for Codeforces dataset items.
        
        Returns:
            List of required fields for processing.
        """
        return ["name", "description", "input_spec", "output_spec", "samples"]
    
    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a DatasetEntry from a Codeforces problem.
        
        Args:
            item: Raw dataset item containing problem details.
                 
        Returns:
            DatasetEntry with the problem as query and test cases in metadata.
        """
        # Apply filters
        if self.limit and self.problem_count >= self.limit:
            return []  # Skip if we've reached the limit
            
        difficulty = item.get("rating")
        if self.difficulty_range and difficulty:
            min_diff, max_diff = self.difficulty_range
            if difficulty < min_diff or difficulty > max_diff:
                return []  # Skip if outside difficulty range
                
        problem_tags = item.get("tags", [])
        if self.tags and not any(tag in problem_tags for tag in self.tags):
            return []  # Skip if no matching tags
        
        # Extract problem components
        name = str(item["name"])
        description = str(item["description"])
        input_spec = str(item["input_spec"])
        output_spec = str(item["output_spec"])
        samples = item["samples"]
        
        # Format the complete problem statement
        problem_statement = f"""# {name}

## Problem Description
{description}

## Input Specification
{input_spec}

## Output Specification
{output_spec}

## Examples
"""
        # Add sample test cases
        for i, sample in enumerate(samples, 1):
            problem_statement += f"""
### Sample {i}
#### Input:
```
{sample['input']}
```

#### Output:
```
{sample['output']}
```
"""
        
        # Increment counter for limit tracking
        self.problem_count += 1
        
        return [
            DatasetEntry(
                query=problem_statement,
                choices={},  # No choices for code problems
                metadata={
                    "name": name,
                    "difficulty": difficulty,
                    "tags": problem_tags,
                    "problem_id": item.get("problem_id", ""),
                    "test_cases": samples,
                }
            )
        ]


# Register the Codeforces dataset
register_metadata(
    name="codeforces",
    description="Competitive programming problems from Codeforces",
    source="open-r1/codeforces",
    task_type=TaskType.CODE_COMPLETION,
    prepper_class=CodeForcesPrepper,
)
```

### 6.4. Custom Evaluator for Codeforces

```python
import json
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator


class CodeCompetitionEvaluator(IEvaluator[str, Dict[str, Any]]):
    """Evaluator for competitive programming problems.
    
    Executes submitted code against multiple test cases in a controlled
    environment, with time limits and memory monitoring.
    """
    
    def __init__(
        self, 
        time_limit: float = 2.0, 
        memory_limit_mb: int = 512,
        languages: Optional[List[str]] = None
    ) -> None:
        """Initialize the code competition evaluator.
        
        Args:
            time_limit: Maximum execution time per test case (seconds)
            memory_limit_mb: Maximum memory usage allowed (MB)
            languages: List of supported languages, defaults to ["python"]
        """
        self.time_limit = time_limit
        self.memory_limit_mb = memory_limit_mb
        self.languages = languages or ["python"]
        
    def evaluate(
        self, system_output: str, correct_answer: Dict[str, Any], **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate generated code against test cases.
        
        Args:
            system_output: Generated code solution
            correct_answer: Dictionary containing test cases and metadata
            **kwargs: Additional parameters including language
        
        Returns:
            EvaluationResult with test case results and execution metrics
        """
        language = kwargs.get("language", "python")
        if language not in self.languages:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": f"Unsupported language: {language}"}
            )
            
        test_cases = correct_answer.get("test_cases", [])
        if not test_cases:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                metadata={"error": "No test cases available"}
            )
            
        # Create a temporary directory for code execution
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the code to a file
            code_file = os.path.join(temp_dir, f"solution.{self._get_extension(language)}")
            with open(code_file, "w") as f:
                f.write(system_output)
                
            # Run each test case
            test_results = []
            passed_count = 0
            
            for i, test in enumerate(test_cases):
                test_input = test.get("input", "")
                expected_output = test.get("output", "").strip()
                
                result = self._run_test_case(
                    code_file=code_file,
                    language=language,
                    test_input=test_input,
                    expected_output=expected_output
                )
                
                if result["passed"]:
                    passed_count += 1
                    
                test_results.append({
                    "test_index": i,
                    "passed": result["passed"],
                    "execution_time": result["execution_time"],
                    "memory_used_mb": result.get("memory_used_mb"),
                    "error": result.get("error"),
                })
                
        # Calculate overall score based on passed test cases
        total_tests = len(test_cases)
        score = passed_count / total_tests if total_tests > 0 else 0.0
        is_correct = passed_count == total_tests
        
        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            metadata={
                "test_results": test_results,
                "passed_count": passed_count,
                "total_tests": total_tests,
            }
        )
        
    def _get_extension(self, language: str) -> str:
        """Get file extension for language.
        
        Args:
            language: Programming language
            
        Returns:
            File extension string
        """
        extensions = {
            "python": "py",
            "c++": "cpp",
            "java": "java",
        }
        return extensions.get(language, "txt")
        
    def _run_test_case(
        self,
        code_file: str,
        language: str,
        test_input: str,
        expected_output: str
    ) -> Dict[str, Any]:
        """Run a single test case for the submitted code.
        
        Args:
            code_file: Path to the source code file
            language: Programming language
            test_input: Input for the test case
            expected_output: Expected output
            
        Returns:
            Dictionary with test results
        """
        command = self._get_run_command(language, code_file)
        
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            stdout, stderr = process.communicate(
                input=test_input, 
                timeout=self.time_limit
            )
            
            execution_time = time.time() - start_time
            
            # Compare output (ignoring whitespace differences)
            actual_output = stdout.strip()
            passed = actual_output == expected_output
            
            return {
                "passed": passed,
                "execution_time": execution_time,
                "actual_output": actual_output,
                "stderr": stderr,
                "exit_code": process.returncode,
            }
            
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "execution_time": self.time_limit,
                "error": "Time limit exceeded",
            }
        except Exception as e:
            return {
                "passed": False,
                "execution_time": 0,
                "error": str(e),
            }
            
    def _get_run_command(self, language: str, code_file: str) -> List[str]:
        """Get the command to run code in the specified language.
        
        Args:
            language: Programming language
            code_file: Path to source code
            
        Returns:
            Command list for subprocess
        """
        if language == "python":
            return ["python", code_file]
        elif language == "c++":
            executable = code_file.replace(".cpp", ".exe")
            return ["g++", code_file, "-o", executable, "&&", executable]
        elif language == "java":
            return ["java", code_file]
        else:
            raise ValueError(f"Unsupported language: {language}")
```

## 7. Integration Examples

### 7.1. Basic Usage Example

```python
from ember.api import datasets, models

# Load and explore AIME dataset
aime_data = datasets("aime")
print(f"AIME dataset size: {len(aime_data)}")

# Get first AIME problem
aime_problem = aime_data[0]
print(f"Problem: {aime_problem.query}")

# Test a model on an AIME problem
model = models.anthropic.claude35_sonnet()
response = model(aime_problem.query)

# Evaluate the response with the proper evaluator
from ember.core.utils.eval.evaluators import NumericToleranceEvaluator
evaluator = NumericToleranceEvaluator()
result = evaluator.evaluate(response, aime_problem.metadata["correct_answer"])
print(f"Evaluation result: {result}")
```

### 7.2. Advanced Usage with Builder Pattern

```python
from ember.api import DatasetBuilder, EvaluationPipeline, models

# Create a dataset with filtered problems
gpqa_dataset = (
    DatasetBuilder()
    .from_registry("gpqa")  # Use registered GPQA dataset
    .sample(10)  # Get 10 random problems
    .transform(lambda entry: {
        **entry,
        "query": f"Choose the correct answer to this PhD-level question:\n\n{entry['query']}"
    })
    .build()
)

# Create an evaluation pipeline
pipeline = EvaluationPipeline()
pipeline.add_model(models.openai.gpt4o())
pipeline.add_model(models.anthropic.claude35_sonnet())

# Run evaluation
results = pipeline.evaluate(gpqa_dataset)
print(results.summary())
```

## 8. Documentation

### 8.1. README.md for Datasets

```markdown
# Ember Datasets

Ember provides access to a diverse set of evaluation datasets through a unified interface. Here's how to use the recently added datasets:

## AIME 2024 (Competition Math)

The AIME dataset contains 30 challenging problems from the American Invitational Mathematics Examination.

```python
# Load AIME dataset (all problems from 2024)
from ember.api import datasets
aime_data = datasets("aime")

# Load only AIME I contest problems
from ember.api import DatasetBuilder
aime_data = DatasetBuilder().from_registry("aime").configure(contest="I").build()
```

## GPQA Diamond (PhD-Level Science)

GPQA Diamond contains 198 graduate-level multiple-choice science questions spanning physics, chemistry, and biology.

```python
# Load GPQA Diamond dataset
gpqa_data = datasets("gpqa")

# Filter by subject area
gpqa_physics = DatasetBuilder().from_registry("gpqa").filter(
    lambda entry: "physics" in entry.metadata.get("subject", "").lower()
).build()
```

## Codeforces (Competitive Programming)

The Codeforces dataset contains competitive programming problems from the Codeforces platform.

```python
# Load Codeforces problems with difficulty range 800-1200
codeforces_data = DatasetBuilder().from_registry("codeforces").configure(
    difficulty_range=(800, 1200), limit=10
).build()
```
```

## 9. Code Style and Engineering Quality

Throughout implementation, we'll adhere to the highest code quality standards:

1. **Clean, Minimalist Code**
   - Precise, concise implementations
   - No unnecessary abstractions
   - Efficient algorithms and data structures

2. **Type Safety**
   - Comprehensive type annotations
   - Proper use of generics
   - Validation of inputs and outputs

3. **Documentation**
   - Thorough, accurate docstrings
   - Purpose, behavior, parameters, returns, exceptions
   - Clear examples

4. **Error Handling**
   - Specific, actionable error messages
   - Graceful degradation
   - Precise exception types

5. **Testing**
   - Comprehensive unit tests
   - Integration tests
   - Property-based tests for evaluators

## 10. Timeline and Milestones

1. **Day 1-2: Core Implementation**
   - Create and implement skeleton classes
   - Set up registration system
   - Implement dataset loaders

2. **Day 3-4: Evaluators and Testing**
   - Implement specialized evaluators
   - Create test suite
   - Performance optimization

3. **Day 5: Integration and Documentation**
   - End-to-end integration
   - Documentation
   - Example notebooks

## 11. Future Extensions

1. **Enhanced Evaluation Metrics**
   - Partial credit for AIME problems
   - Runtime efficiency scoring for Codeforces
   - Domain-specific analysis for GPQA

2. **UI Integration**
   - Interactive visualization of results
   - Leaderboard comparisons

3. **Expanded Dataset Coverage**
   - Additional AIME years
   - GPQA full dataset
   - More competitive programming platforms

## 12. Implementation Progress

### Implementation Checklist

- [x] **Setup & Structure**
  - [x] Create directory structure
  - [x] Set up test fixtures
  - [x] Create skeleton files

- [x] **AIME 2024**
  - [x] Implement AIMEConfig
  - [x] Implement AIMEPrepper (fully)
  - [x] Create AIMEAnswerEvaluator
  - [x] Write unit tests
  - [x] Create standalone test for validation

- [ ] **GPQA Diamond**
  - [x] Implement GPQAConfig (skeleton)
  - [ ] Implement GPQAPrepper (fully)
  - [ ] Configure MCQ evaluator
  - [x] Write unit tests (skeleton)

- [ ] **Codeforces**
  - [x] Implement CodeForcesConfig (skeleton)
  - [ ] Implement CodeForcesPrepper (fully)
  - [x] Create CodeCompetitionEvaluator (skeleton)
  - [ ] Implement sandboxed execution
  - [x] Write unit tests (skeleton)

- [x] **Documentation**
  - [x] Create comprehensive guide for adding datasets
  - [x] Document architectural patterns
  - [x] Include integration instructions
  - [x] Create example usage code

- [ ] **Final Integration**
  - [ ] Update central registry
  - [ ] Add evaluator exports
  - [ ] End-to-end validation
  - [ ] Performance testing

### Revised Implementation Strategy

We've updated our approach based on a deeper understanding of Ember's architecture:

1. **Dataset Implementation**
   - Implement dataset prepper classes with full functionality following Ember patterns
   - Keep all implementation focused only on data transformation
   - Remove any direct registration code from dataset files

2. **Central Registration**
   - Update `initialize_registry()` in `registry.py` to include our new datasets
   - Add proper imports for all dataset modules
   - Follow existing patterns for registration using `UNIFIED_REGISTRY.register_metadata()`

3. **Evaluators Implementation**
   - Implement evaluators for each dataset type
   - Integration with existing evaluator interfaces
   - Unit tests for evaluator correctness

4. **Documentation**
   - Added comprehensive guide on adding new datasets
   - Document architectural patterns and best practices
   - Create example usage notebooks

5. **Testing Strategy**
   - Comprehensive unit tests for prepper classes
   - Integration tests verifying end-to-end functionality
   - Performance testing for large datasets