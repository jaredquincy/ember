from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol
import numpy as np

from ember.core.registry.model.providers.provider_capability import (EmbeddingProviderModel,
                                                                     EmbeddingResponse)
from ember.core.registry.model.providers.openai.openai_provider import create_openai_embedding_model

################################################################
# 1) Embedding Model Interfaces & Implementations
################################################################

# NOTE: These protocols are now outdated by the EmbeddingProviderModel/CapabilityModel interfaces

class EmbeddingModel(Protocol):
    """Interface for embedding models.

    This protocol defines the minimal interface required to compute a text
    embedding. Implementations may use local models, external APIs, or custom
    neural networks.

    Methods:
        embed_text: Compute the embedding for a given text.
    """

    def embed_text(self, text: str) -> List[float]:
        """Computes the embedding vector for the provided text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        ...


class MockEmbeddingModel:
    """Mock implementation of an embedding model using naive ASCII encoding.

    This simple model converts each character in the text to a normalized ASCII
    value. It is intended solely for demonstration and testing purposes.

    Methods:
        embed_text: Converts text to a sequence of normalized ASCII values.
    """

    def embed_text(self, text: str) -> List[float]:
        """Embeds text by converting each character to its normalized ASCII code.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: A list of floats representing the embedding. Returns an
            empty list if the text is empty.
        """
        if not text:
            return []
        return [ord(ch) / 256.0 for ch in text]

################################################################
# 2) Similarity Metric Interface & Implementations
################################################################


class SimilarityMetric(ABC):
    """Abstract base class for computing similarity between embedding vectors.

    Subclasses must implement the similarity method to calculate a similarity
    score between two vectors.
    """

    @abstractmethod
    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates the similarity between two embedding vectors.

        Args:
            vec_a (List[float]): The first embedding vector.
            vec_b (List[float]): The second embedding vector.

        Returns:
            float: The similarity score, typically in the range [0, 1] or [-1, 1].
        """
        ...


class CosineSimilarity(SimilarityMetric):
    """Implementation of cosine similarity for embedding vectors.

    The cosine similarity is defined as:
        similarity(a, b) = (a · b) / (||a|| * ||b||)

    Returns 0.0 if either vector is empty or if any vector's norm is zero.
    """

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two embedding vectors.

        Args:
            vec_a (List[float]): The first embedding vector.
            vec_b (List[float]): The second embedding vector.

        Returns:
            float: The cosine similarity score.
        """
        if not vec_a or not vec_b:
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))


################################################################
# 3) High-Level Utility Function
################################################################


def calculate_text_similarity(
    text1: str, text2: str, model: EmbeddingProviderModel, metric: SimilarityMetric
) -> float:
    """Calculates text similarity using an embedding model and a similarity metric.

    This function generates embeddings for the provided texts and then computes a
    similarity score using the given similarity metric.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        model (EmbeddingModel): An instance conforming to the embedding model interface.
        metric (SimilarityMetric): An instance implementing a similarity metric.

    Returns:
        float: The computed similarity score.
    """
    response1: EmbeddingResponse = model.embed_text(input_text=text1)
    response2: EmbeddingResponse = model.embed_text(input_text=text2)
    
    embeddings1: List[float] = response1.embeddings
    embeddings2: List[float] = response2.embeddings

    return metric.similarity(vec_a=embeddings1, 
                             vec_b=embeddings2)


################################################################
# 4) Example Usage (Executable as Script)
################################################################
if __name__ == "__main__":
    openai_embedding_model = create_openai_embedding_model()
    cosine_simlarity = CosineSimilarity()

    text_a: str = "Hello world!"
    text_b: str = "Hello, world??"

    score: float = calculate_text_similarity(
        text1=text_a, text2=text_b, model=openai_embedding_model, metric=cosine_simlarity
    )
    print(f"Similarity between '{text_a}' and '{text_b}': {score}")
