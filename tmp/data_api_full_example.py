"""Full data API example demonstrating the pattern from the README.

This script implements the data handling example shown in the Ember README.md.
We use our standalone_data module to avoid the circular import issues.
"""

import random
from typing import Any, Dict, List, Callable
from pydantic import BaseModel, Field

# Import from our standalone version
from standalone_data import Dataset, DatasetEntry, DatasetInfo, TaskType


class DatasetBuilder:
    """Builder pattern implementation for datasets as shown in the README."""

    def __init__(self):
        """Initialize a new dataset builder with default values."""
        self._split = None
        self._sample_size = None
        self._seed = None
        self._transform_fn = None
        self._subset = None

    def from_registry(self, dataset_name: str):
        """Select a dataset from the registry (simulated).

        Args:
            dataset_name: Name of the dataset

        Returns:
            Self for method chaining
        """
        self._dataset_name = dataset_name
        return self

    def subset(self, subset_name: str):
        """Select a specific subset of the dataset.

        Args:
            subset_name: Name of the subset

        Returns:
            Self for method chaining
        """
        self._subset = subset_name
        return self

    def split(self, split_name: str):
        """Set the dataset split to load.

        Args:
            split_name: Name of the split (e.g., "train", "test", "validation")

        Returns:
            Self for method chaining
        """
        self._split = split_name
        return self

    def sample(self, count: int):
        """Set the number of samples to load.

        Args:
            count: Number of samples to load

        Returns:
            Self for method chaining
        """
        self._sample_size = count
        return self

    def transform(self, transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Apply a transformation to each dataset entry.

        Args:
            transform_fn: Function to transform entries

        Returns:
            Self for method chaining
        """
        self._transform_fn = transform_fn
        return self

    def build(self, dataset_name: str = None):
        """Build and load the dataset with the configured parameters.

        Args:
            dataset_name: Optional name of the dataset (overrides from_registry)

        Returns:
            Loaded dataset
        """
        if dataset_name:
            self._dataset_name = dataset_name

        if not hasattr(self, "_dataset_name"):
            raise ValueError(
                "Dataset name must be provided via from_registry() or build()"
            )

        # In a real implementation, this would load the actual dataset
        # For this example, we create a mock dataset based on the name

        # Set random seed if provided
        if self._seed is not None:
            random.seed(self._seed)

        # Create mock dataset based on name
        if self._dataset_name.lower() == "mmlu":
            dataset = self._create_mmlu_dataset()
        else:
            # Generic dataset
            dataset = self._create_generic_dataset()

        # Apply transformations if specified
        if self._transform_fn and callable(self._transform_fn):
            transformed_entries = []
            for entry in dataset.entries:
                # Convert to dict, apply transform, then back to DatasetEntry
                entry_dict = entry.model_dump()
                transformed = self._transform_fn(entry_dict)
                transformed_entries.append(DatasetEntry(**transformed))
            dataset.entries = transformed_entries

        # Apply sampling if specified
        if self._sample_size is not None:
            count = min(self._sample_size, len(dataset.entries))
            dataset.entries = random.sample(dataset.entries, count)

        return dataset

    def _create_mmlu_dataset(self):
        """Create a mock MMLU dataset."""
        subjects = {
            "physics": [
                (
                    "What is Newton's first law?",
                    {
                        "A": "Objects in motion stay in motion",
                        "B": "Force equals mass times acceleration",
                        "C": "Energy is conserved",
                        "D": "Matter cannot be created or destroyed",
                    },
                    "A",
                ),
                (
                    "What particle is exchanged in electromagnetic interactions?",
                    {"A": "Graviton", "B": "Photon", "C": "Gluon", "D": "W boson"},
                    "B",
                ),
                (
                    "What is the SI unit of electric current?",
                    {"A": "Volt", "B": "Ohm", "C": "Ampere", "D": "Coulomb"},
                    "C",
                ),
                (
                    "Which scientist proposed the theory of general relativity?",
                    {
                        "A": "Isaac Newton",
                        "B": "Niels Bohr",
                        "C": "Max Planck",
                        "D": "Albert Einstein",
                    },
                    "D",
                ),
                (
                    "What is the speed of light in vacuum?",
                    {
                        "A": "3 × 10^8 m/s",
                        "B": "3 × 10^6 m/s",
                        "C": "3 × 10^10 m/s",
                        "D": "3 × 10^4 m/s",
                    },
                    "A",
                ),
            ],
            "mathematics": [
                (
                    "What is the derivative of sin(x)?",
                    {"A": "cos(x)", "B": "-sin(x)", "C": "tan(x)", "D": "-cos(x)"},
                    "A",
                ),
                (
                    "What is the value of π (pi) to two decimal places?",
                    {"A": "3.41", "B": "3.14", "C": "3.12", "D": "3.18"},
                    "B",
                ),
                (
                    "What is the formula for the area of a circle?",
                    {"A": "πr", "B": "2πr", "C": "πr²", "D": "πr³"},
                    "C",
                ),
                ("What is 7 × 8?", {"A": "54", "B": "65", "C": "49", "D": "56"}, "D"),
                (
                    "What is the square root of 144?",
                    {"A": "12", "B": "14", "C": "16", "D": "10"},
                    "A",
                ),
            ],
        }

        # Select subset if specified
        if self._subset and self._subset in subjects:
            questions = subjects[self._subset]
        else:
            # Combine all subjects
            questions = []
            for subject_questions in subjects.values():
                questions.extend(subject_questions)

        # Create entries
        entries = []
        for i, (question, choices, answer) in enumerate(questions):
            subject = self._subset or next(
                (s for s, qs in subjects.items() if (question, choices, answer) in qs),
                "unknown",
            )
            entry = DatasetEntry(
                query=question,
                choices=choices,
                metadata={
                    "correct_answer": answer,
                    "category": subject,
                    "id": f"mmlu_{subject}_{i}",
                },
            )
            entries.append(entry)

        # Create dataset info
        info = DatasetInfo(
            name="mmlu",
            description="Massive Multitask Language Understanding benchmark",
            source="cais/mmlu",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        return Dataset(entries=entries, info=info)

    def _create_generic_dataset(self):
        """Create a generic dataset."""
        entries = []
        for i in range(10):
            entry = DatasetEntry(
                query=f"Generic question {i}?",
                choices={
                    "A": f"Option A for question {i}",
                    "B": f"Option B for question {i}",
                    "C": f"Option C for question {i}",
                    "D": f"Option D for question {i}",
                },
                metadata={
                    "correct_answer": random.choice(["A", "B", "C", "D"]),
                    "id": f"generic_{i}",
                },
            )
            entries.append(entry)

        # Create dataset info
        info = DatasetInfo(
            name=self._dataset_name,
            description=f"Generic {self._dataset_name} dataset",
            source="local",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        return Dataset(entries=entries, info=info)


class Evaluator:
    """Evaluator for measuring model performance."""

    @classmethod
    def from_registry(cls, evaluator_name: str):
        """Create an evaluator from the registry.

        Args:
            evaluator_name: Name of the evaluator

        Returns:
            Configured evaluator
        """
        if evaluator_name == "accuracy":
            return cls(metric="accuracy")
        elif evaluator_name == "response_quality":
            return cls(metric="response_quality")
        else:
            return cls(metric=evaluator_name)

    @classmethod
    def from_function(cls, eval_fn: Callable):
        """Create an evaluator from a custom function.

        Args:
            eval_fn: Evaluation function

        Returns:
            Configured evaluator
        """
        return cls(metric="custom", eval_fn=eval_fn)

    def __init__(self, metric: str, eval_fn: Callable = None):
        """Initialize evaluator.

        Args:
            metric: Name of the metric
            eval_fn: Optional custom evaluation function
        """
        self.metric = metric
        self.eval_fn = eval_fn

    def evaluate(self, prediction: Any, reference: Any) -> Dict[str, float]:
        """Evaluate a prediction against a reference.

        Args:
            prediction: Model prediction
            reference: Ground truth

        Returns:
            Evaluation metrics
        """
        if self.eval_fn:
            return self.eval_fn(prediction, reference)

        # Simple built-in metrics
        if self.metric == "accuracy":
            # For multiple-choice questions
            if isinstance(prediction, str) and isinstance(reference, str):
                score = 1.0 if prediction == reference else 0.0
                return {"accuracy": score}
            return {"accuracy": 0.0}

        elif self.metric == "response_quality":
            # Mock quality metric
            if isinstance(prediction, str) and isinstance(reference, str):
                # Simple length-based quality score for demonstration
                pred_len = len(prediction)
                ref_len = len(reference)
                ratio = min(pred_len, ref_len) / max(pred_len, ref_len)
                return {"response_quality": ratio}
            return {"response_quality": 0.0}

        return {self.metric: 0.5}  # Default placeholder value


class EvaluationPipeline:
    """Evaluation pipeline for measuring model performance."""

    def __init__(self, evaluators: List[Evaluator]):
        """Initialize evaluation pipeline.

        Args:
            evaluators: List of evaluators to use
        """
        self.evaluators = evaluators

    def evaluate(self, model: Any, dataset: Dataset) -> Dict[str, Any]:
        """Evaluate a model on a dataset.

        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on

        Returns:
            Evaluation results
        """
        # In a real implementation, this would run the model on each dataset entry
        # and calculate metrics using the evaluators

        # For this example, we simulate model outputs and evaluations
        results = {}

        # Initialize metric aggregates
        for evaluator in self.evaluators:
            if evaluator.metric == "custom":
                # For custom evaluators, we don't know the metric names in advance
                # Results will be added when the evaluator is called
                pass
            else:
                results[evaluator.metric] = 0.0

        # Process each entry
        for entry in dataset:
            # Simulate model prediction
            # In real usage, this would call the actual model
            if hasattr(model, "predict"):
                prediction = model.predict(entry.query)
            else:
                # Mock prediction for demonstration
                choices = list(entry.choices.values())
                correct_idx = ord(entry.metadata.get("correct_answer", "A")) - ord("A")
                # 80% chance of being correct for demonstration
                if random.random() < 0.8:
                    prediction = choices[correct_idx]
                else:
                    wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
                    prediction = choices[random.choice(wrong_indices)]

            # Get correct answer
            correct_letter = entry.metadata.get("correct_answer", "A")
            correct_answer = entry.choices.get(correct_letter, "")

            # Calculate metrics for this entry
            for evaluator in self.evaluators:
                entry_metrics = evaluator.evaluate(prediction, correct_answer)

                # Add metrics to results
                for metric_name, value in entry_metrics.items():
                    if metric_name not in results:
                        results[metric_name] = value
                    else:
                        results[metric_name] += value

        # Average metrics across entries
        for metric_name in results:
            if len(dataset) > 0:
                results[metric_name] /= len(dataset)

        return results


def score_factual_content(prediction: str, reference: str) -> Dict[str, float]:
    """Simulate a custom evaluation function for factual accuracy."""
    # This would normally be a more sophisticated comparison
    # For demonstration, we use a simple word overlap metric
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())

    if not pred_words or not ref_words:
        return {"factual_accuracy": 0.0}

    # Jaccard similarity
    intersection = len(pred_words.intersection(ref_words))
    union = len(pred_words.union(ref_words))

    return {"factual_accuracy": intersection / union if union > 0 else 0.0}


class MockModel:
    """Mock model that simulates responses."""

    def predict(self, query: str) -> str:
        """Generate a prediction for a query.

        Args:
            query: Input query

        Returns:
            Model prediction
        """
        # Very simple mock implementation that just returns a response based on keywords
        if "capital" in query.lower():
            return "Paris is the capital of France."
        elif "mammal" in query.lower():
            return "The dolphin is a mammal because it breathes air, has hair, and produces milk."
        elif "square root" in query.lower():
            return "The square root of 144 is 12."
        else:
            return f"Response to: {query}"


def main():
    """Run the data API full example."""
    print("Ember Data API Example from README")
    print("=================================")

    # Create a dataset using the builder pattern as shown in the README
    print("\n1. Loading a dataset with the builder pattern:")
    dataset = (
        DatasetBuilder()
        .from_registry("mmlu")  # Use a registered dataset
        .subset("physics")  # Select a specific subset
        .split("test")  # Choose the test split
        .sample(3)  # Random sample of 3 items
        .transform(  # Apply transformations
            lambda x: {
                "query": f"Question: {x['query']}",
                "choices": x["choices"],
                "metadata": x["metadata"],
            }
        )
        .build()
    )

    print(f"Loaded dataset with {len(dataset)} entries and info: {dataset.info}")
    for i, entry in enumerate(dataset.entries):
        print(f"\nEntry {i+1}:")
        print(f"Query: {entry.query}")
        print(f"Correct answer: {entry.metadata['correct_answer']}")

    # Create a model
    model = MockModel()

    # Create an evaluation pipeline
    print("\n2. Creating an evaluation pipeline:")
    eval_pipeline = EvaluationPipeline(
        [
            # Standard metrics
            Evaluator.from_registry("accuracy"),
            Evaluator.from_registry("response_quality"),
            # Custom evaluation metric
            Evaluator.from_function(
                lambda prediction, reference: {
                    "factual_accuracy": score_factual_content(prediction, reference)[
                        "factual_accuracy"
                    ]
                }
            ),
        ]
    )

    # Evaluate the model
    print("\n3. Evaluating the model:")
    results = eval_pipeline.evaluate(model, dataset)

    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")


if __name__ == "__main__":
    main()
