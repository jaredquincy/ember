from __future__ import annotations

from typing import Any, List, Tuple

from .base_evaluator import EvaluationResult, IEvaluator

# diversity imports
from diversity import compression_ratio
import Levenshtein
import numpy as np
from ember.core.utils.embedding_utils import (CosineSimilarity, 
                                              calculate_text_similarity)
from ember.core.registry.model.providers.provider_capability import EmbeddingProviderModel
from ember.core.registry.model.providers.openai.openai_provider import create_openai_embedding_model

import logging

# Composite Evaluator Example
class DiversityEnsembledEvaluator(IEvaluator[List[str], None]):
    """Evaluator that combines multiple diversity metrics to assess ensemble output diversity.

    Computes diversity as an average of cosine similarity, compression ratio, and edit distance.
    The higher this score is, the more diverse your text.

    Args:
        system_output (List[str]): List of generated outputs from the system.
        embedding_model (EmbeddingModel): The embedding model to compute cosine similarity.

    Returns:
        EvaluationResult: Average of the three diversity scores with `is_correct=True`.
    """
    def __init__(self, embedding_model: EmbeddingProviderModel):
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            logging.warning("DiversityEnsembledEvaluator isn't initialized with an embedding model")

    def evaluate(
        self,
        system_output: List[str],
        **kwargs
    ) -> EvaluationResult:
        if not system_output:
            logging.debug("DiversityEnsembledEvaluator didn't receive an output")
            return EvaluationResult(is_correct=False, score=-1)
        if self.embedding_model is None:
            logging.debug("DiversityEnsembledEvaluator wasn't initialized with an embedding model")
            return EvaluationResult(is_correct=False, score=-1)
        if len(system_output) == 1:
            logging.debug("DiversityEnsembledEvaluator only received one string of text")
            return EvaluationResult(is_correct=True, score=0)

        # Lower cosine similarity --> more diverse
        cosine_score = 1.0 - DiversityCosineSimilarityEvaluator(embedding_model=self.embedding_model).evaluate(system_output).score
        # higher compression score --> more diverse
        compression_score = DiversityCompressionEvaluator().evaluate(system_output).score
        # higher edit distance --> more diverse
        edit_score = DiversityEditDistanceEvaluator().evaluate(system_output).score

        avg_diversity = (cosine_score + compression_score + edit_score) / 3

        return EvaluationResult(
            is_correct=True,
            score=avg_diversity,
            metadata={"responses": system_output}
        )


class DiversityCosineSimilarityEvaluator(IEvaluator[List[str], None]):
    """Evaluator that computes average pairwise cosine similarity between outputs.

    Lower average cosine similarity implies greater semantic diversity.

    Args:
        system_output (List[str]): List of generated outputs from the system.
        embedding_model (EmbeddingModel): The embedding model used to compute cosine similarity.

    Returns:
        EvaluationResult: Result with average similarity score and output metadata.
    """
    def __init__(self, embedding_model: EmbeddingProviderModel = None):
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            logging.warning("DiversityCosineEvaluator isn't initialized with an embedding model " + 
                            "Using default OpenAI embedding model instead")
            self.embedding_model = create_openai_embedding_model()

    def evaluate(
        self,
        system_output: List[str],
        **kwargs
    ) -> EvaluationResult:
        if not system_output:
            logging.debug("DiversityCosineEvaluator didn't receive an output")
            return EvaluationResult(is_correct=False, score=-1)
        if self.embedding_model is None:
            logging.debug("DiversityCosineEvaluator wasn't initialized with an embedding model")
            return EvaluationResult(is_correct=False, score=-1)
        if len(system_output) == 1:
            logging.deubg("DiversityCosineEvaluator only received one string of text")
            return EvaluationResult(is_correct=True, score=0)

        cosine_similarity = CosineSimilarity()
        scores = []

        # TODO IDEA: Compute embedding vectors for all system_output --> get the average
        # Then compute cosine similarity between all other outputs

        # Compare every possible combination of system_output vectors
        for i in range(len(system_output)):
            for j in range(i + 1, len(system_output)):
                sim = calculate_text_similarity(
                    system_output[i], system_output[j], self.embedding_model, metric=cosine_similarity
                )
                scores.append(sim)

        avg_score = float(np.average(scores))

        return EvaluationResult(
            is_correct=True,
            score=avg_score,
            metadata={"responses": system_output}
        )


class DiversityCompressionEvaluator(IEvaluator[List[str], None]):
    """Evaluator that measures diversity using a compression ratio heuristic.

    Lower compression ratio indicates higher textual diversity. The final score is scaled
    based on a minimum number of responses (5) and minimum total character count (100).

    Args:
        system_output (List[str]): List of generated responses.

    Returns:
        EvaluationResult: Scaled diversity score based on compression.
    """

    def evaluate(
        self,
        system_output: List[str],
        **kwargs
    ) -> EvaluationResult:
        if not system_output:
            return EvaluationResult(is_correct=False, score=-1)

        total_chars = sum(len(r) for r in system_output)
        # ratio = (size of compressed data) / (size of uncompressed data)
        # Higher ratio is --> more diverse
        ratio = 1 / compression_ratio(system_output)
        # Penalize inputs with few words (hard to measure) and inputs with very few characters
        # Note that this is a temporary patch for compression_ratio does not normalizing over word length
        scaled_score = ratio * min(1, len(system_output) / 5) * min(1, total_chars / 100)

        return EvaluationResult(
            is_correct=True,
            score=scaled_score,
            metadata={"responses": system_output}
        )


class DiversityEditDistanceEvaluator:
    """Evaluator that measures lexical diversity using normalized Levenshtein edit distance.

    Computes average pairwise normalized edit distance across all outputs.

    Args:
        system_output (List[str]): List of generated responses.

    Returns:
        EvaluationResult: Average normalized edit distance score.
    """

    def evaluate(self, system_output: List[str], **kwargs) -> EvaluationResult:
        if not system_output:
            return EvaluationResult(is_correct=False, score=-1, metadata={})

        score = self.compute_distance(system_output)

        return EvaluationResult(
            is_correct=True,
            score=score,
            metadata={"responses": system_output}
        )

    def compute_distance(self, outputs: List[str]) -> float:
        n = len(outputs)
        if n < 2:
            return 0.0

        total_distance = 0.0
        num_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist = Levenshtein.distance(outputs[i], outputs[j])
                max_len = max(len(outputs[i]), len(outputs[j]))
                norm_dist = dist / max_len if max_len > 0 else 0
                total_distance += norm_dist
                num_pairs += 1

        return total_distance / num_pairs if num_pairs > 0 else 0.0


class DiversityNoveltyEvaluator:
    """Evaluator that measures novelty of each output relative to previously generated ones.

    For each response, computes its cosine distance from all prior responses.
    Higher novelty implies lower similarity to prior outputs.

    Args:
        model (EmbeddingModel): Embedding model used for computing cosine similarity.
        system_output (List[str]): List of outputs ordered by generation.

    Returns:
        EvaluationResult: Average novelty score across the sequence.
    """

    def __init__(self, embedding_model: EmbeddingProviderModel = None):
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            logging.warning("DiversityNoveltyEvaluator isn't initialized with an embedding model " + 
                            "Using default OpenAI embedding model instead")
            self.embedding_model = create_openai_embedding_model()
        
    def evaluate(self,
                system_output: List[str],
                **kwargs
                ) -> EvaluationResult:
        """
        Evaluates the novelty of each response in a sequence relative to the responses that came before it,
        using cosine similarity of embeddings.

        For each response, an embedding is computed and compared against embeddings of all prior responses.
        The novelty score is defined as 1.0 minus the maximum cosine similarity with any prior response.
        A high score indicates a novel response, while a low score indicates redundancy.

        Note:
        - If all responses are identical, the first response gets a score of 1.0 while 
          all others get 0.0, resulting in an average (and minimum) score of 1/len(system_output).

        Returns:
            EvaluationResult:
                - is_correct: True if evaluation ran successfully.
                - score: Average novelty score across all responses.
                - metadata: Contains raw responses and their individual novelty scores.
        """

        if len(system_output) == 0:
            logging.warning("Length of inputs to evaluate function is zero")
            return EvaluationResult(is_correct=False, score=-1, metadata={})
        
        novelty_scores = []
        prior_embeddings = []

        for r in system_output:
            new_emb, novelty = self._compute_novelty(self.embedding_model, r, prior_embeddings)
            novelty_scores.append(novelty)
            prior_embeddings.append(new_emb)

        avg_score = float(np.mean(novelty_scores)) if novelty_scores else 0.0

        return EvaluationResult(
            is_correct=True,
            score=avg_score,
            metadata={
                "responses": system_output,
                "novelty_scores": novelty_scores
            }
        )

    def _compute_novelty(self,
                        model: EmbeddingProviderModel,
                        response: str,
                        prior_embeddings: List[str]
                    ) -> Tuple[np.ndarray, float]:
        
        new_emb = model.embed_text(response).embeddings

        if not prior_embeddings:
            return new_emb, 1.0

        similarities = [
            np.dot(new_emb, pe) / (np.linalg.norm(new_emb) * np.linalg.norm(pe))
            for pe in prior_embeddings
        ]

        return new_emb, 1.0 - max(similarities)


if __name__ == "__main__":
    text_embedding_ada_002 = create_openai_embedding_model("text-embedding-ada-002")

    # List of text that represents completely n
    very_diverse_text = ["Bananas don't belong in briefcases, but socks and t-shirts do!", 
                        "Abraham Lincoln was the 16th president of the United States of America", 
                        "ERROR 404: Index Not Found"]

    # This group of text all rephrase the same request, except 
    different_words_not_diverse_strs = ["Could you please lend me a hand with this?", 
                                        "Might you assist me with a task?", 
                                        "Can you spare a second to help me do something?"]

    repetition_strs = ["This is a sample text with lots of repetition.", 
                    "This is a sample text with lots of repetition.",
                    "This is a sample text with lots of repetition."]

    # List of sample strings that have varying levels of diversity:
    test_strings = [very_diverse_text, different_words_not_diverse_strs, repetition_strs]


    # Measure Cosine similarity
    cosine_similarity_evaluator = DiversityCosineSimilarityEvaluator(text_embedding_ada_002)

    print("\n" + "=" * 50 )
    print("Cosine Similarity Evaluator\n")
    for i in range(len(test_strings)):
        print(f"Computing cosine-similarity for the following strings: ")
        for j in range(len(test_strings[i])):
            print(f"String {j+1}: {test_strings[i][j]}")
        score: float = cosine_similarity_evaluator.evaluate(system_output=test_strings[i]).score
        print(f"Diversity score: {score}\n")


    # Measure Edit Distance
    print("=" * 50 + "\nEdit Distance Evaluator\n")
    edit_distance_evaluator = DiversityEditDistanceEvaluator()

    for i in range(len(test_strings)):
        print(f"Computing Edit-Distance for the following strings: ")
        for j in range(len(test_strings[i])):
            print(f"String {j+1}: {test_strings[i][j]}")
        score: float = edit_distance_evaluator.evaluate(system_output=test_strings[i]).score
        print(f"Edit-Distance score: {score}\n")
    print("=" * 50 + "\n")


    # Measure Novelty
    print("=" * 50 + "\nNovelty Evaluator\n")
    novelty_evaluator = DiversityNoveltyEvaluator()

    for i in range(len(test_strings)):
        print(f"Computing Novelty for the following strings: ")
        for j in range(len(test_strings[i])):
            print(f"String {j+1}: {test_strings[i][j]}")
        score: float = novelty_evaluator.evaluate(system_output=test_strings[i]).score
        print(f"Novelty score: {score}\n")
    print("=" * 50 + "\n")


    # Measure Compression Ratio
    print("=" * 50 + "\nCompression Ratio Evaluator\n")
    novelty_evaluator = DiversityCompressionEvaluator()

    for i in range(len(test_strings)):
        print(f"Computing Compression Ratio for the following strings: ")
        for j in range(len(test_strings[i])):
            print(f"String {j+1}: {test_strings[i][j]}")
        score: float = novelty_evaluator.evaluate(system_output=test_strings[i]).score
        print(f"Compression Ratio: {score}\n")
    print("=" * 50 + "\n")


    # Measure Ensembled Diversity
    print("=" * 50 + "\nEnsembled Diversity Evaluator\n")
    novelty_evaluator = DiversityCompressionEvaluator()

    for i in range(len(test_strings)):
        print(f"Computing Ensembled Diversity Score for the following strings: ")
        for j in range(len(test_strings[i])):
            print(f"String {j+1}: {test_strings[i][j]}")
        score: float = novelty_evaluator.evaluate(system_output=test_strings[i]).score
        print(f"Ensembled Diversity Score: {score}\n")
    print("=" * 50 + "\n")