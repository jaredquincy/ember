import logging
from typing import Any, ClassVar, Dict, List, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Protocol, TypedDict

from ember.core.registry.model.base.schemas.chat_schemas import (
    ProviderParams,
)
from ember.core.registry.model.base.schemas.usage import UsageStats
from ember.core.registry.model.base.utils.model_registry_exceptions import (
    InvalidPromptError,
    ProviderAPIError,
)
from ember.core.registry.model.providers.base_provider import BaseProviderModel

logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    """Universal text completion request model.

    Similar to ChatRequest but designed for single-turn text completion.
    Used for traditional completion models that predate chat-oriented models.

    Attributes:
        prompt: The text prompt to complete.
        max_tokens: Optional maximum number of tokens to generate.
        temperature: Optional sampling temperature controlling randomness.
        stop_sequences: Optional list of sequences that signal the end of generation.
        provider_params: Provider-specific parameters as a flexible dictionary.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stop_sequences: Optional[List[str]] = None
    provider_params: ProviderParams = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Universal text completion response model.

    Standardizes the response format for text completion models.

    Attributes:
        text: The generated completion text.
        raw_output: The unprocessed provider-specific response data.
        usage: Optional usage statistics for token counting and cost tracking.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    text: str
    raw_output: Any = None
    usage: Optional[UsageStats] = None


class EmbeddingRequest(BaseModel):
    """Request model for generating vector embeddings from text.

    Used to generate semantic vector representations that capture the meaning
    of input text, suitable for similarity comparisons, clustering, and search.

    Attributes:
        input: Text input(s) to embed - can be a single string or list of strings.
        model: Optional specific embedding model to use when the provider has multiple.
        provider_params: Provider-specific parameters as a flexible dictionary.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    input: Union[str, List[str]]
    model: Optional[str] = None
    provider_params: ProviderParams = Field(default_factory=dict)

    @field_validator("input")
    def validate_input(cls, value: Union[str, List[str]]) -> Union[str, List[str]]:
        """Validating the input text is not empty.

        Args:
            value: The input text(s) to validate.

        Returns:
            The validated input value.

        Raises:
            ValueError: If input is empty string or empty list.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError("Input text cannot be empty")
        if isinstance(value, list) and (
            len(value) == 0 or all(not t.strip() for t in value)
        ):
            raise ValueError("Input list cannot be empty or contain only empty strings")
        return value


class EmbeddingResponse(BaseModel):
    """Response model containing vector embeddings.

    Contains numerical vector representations of input text that capture semantic meaning.

    Attributes:
        embeddings: Vector representation(s) of the input text(s).
        model: Name of the embedding model used.
        dimensions: The dimensionality of the embedding vectors.
        raw_output: The unprocessed provider-specific response data.
        usage: Optional usage statistics for token counting and cost tracking.
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # Disable Pydantic's protected namespace checks
    )

    embeddings: Union[List[float], List[List[float]]]
    model: str
    dimensions: int
    raw_output: Any = None
    usage: Optional[UsageStats] = None


# Type variable for implementation-specific typing
ModelT = TypeVar("ModelT", bound="CapabilityModel")

class TextCompletionCapable(Protocol):
    """Protocol defining the interface for text completion models.

    Provider implementations supporting text completion should implement this protocol.
    """

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Processing a text completion request.

        Args:
            request: The text completion request.

        Returns:
            The text completion response.
        """
        ...

    def complete_text(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Convenience method for simple text completion.

        Args:
            prompt: The text to complete.
            **kwargs: Additional parameters for the completion request.

        Returns:
            The text completion response.
        """
        ...


class EmbeddingCapable(Protocol):
    """Protocol defining the interface for embedding models.

    Provider implementations supporting embeddings should implement this protocol.
    """

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generating embeddings for the input text(s).

        Args:
            request: The embedding request.

        Returns:
            The embedding response containing vector representations.
        """
        ...

    def embed_text(
        self, input_text: Union[str, List[str]], **kwargs: Any
    ) -> EmbeddingResponse:
        """Convenience method for simple embedding generation.

        Args:
            input_text: The text(s) to embed.
            **kwargs: Additional parameters for the embedding request.

        Returns:
            The embedding response with vector representations.
        """
        ...


# Base class for capability-aware models
class CapabilityModel(BaseProviderModel):
    """Extended base provider model with capability flags.

    This class extends BaseProviderModel with explicit capability tracking
    to allow runtime capability detection for different model types.

    Attributes:
        CAPABILITIES: Class variable mapping capability names to support flags.
    """

    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": False,
        "embedding": False,
    }


# -----------------------------------------------------------------------------
# PART 3: Extended Provider Base Classes
# -----------------------------------------------------------------------------


class TextCompletionProviderModel(CapabilityModel, TextCompletionCapable):
    """Base class for text completion model providers.

    Extends the BaseProviderModel to support text completion capabilities.
    Providers supporting text completion should inherit from this class.
    """

    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": True,
        "embedding": False,
    }

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Processing a text completion request.

        Args:
            request: The text completion request.

        Returns:
            The text completion response.

        Raises:
            NotImplementedError: If the provider has not implemented this capability.
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support text completion"
        )

    def complete_text(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Convenience method for text completion.

        Creates a CompletionRequest from the prompt and additional parameters,
        then delegates to the complete() method for processing.

        Args:
            prompt: The text to complete.
            **kwargs: Additional parameters for the completion request.

        Returns:
            The text completion response.
        """
        request = CompletionRequest(prompt=prompt, **kwargs)
        return self.complete(request=request)


class EmbeddingProviderModel(CapabilityModel, EmbeddingCapable):
    """Base class for embedding model providers.

    Extends the BaseProviderModel to support embedding capabilities.
    Providers supporting embeddings should inherit from this class.
    """

    CAPABILITIES: ClassVar[Dict[str, bool]] = {
        "chat": True,
        "completion": False,
        "embedding": True,
    }

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generating embeddings for the input text(s).

        Args:
            request: The embedding request.

        Returns:
            The embedding response containing vector representations.

        Raises:
            NotImplementedError: If the provider has not implemented this capability.
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support embeddings"
        )

    def embed_text(
        self, input_text: Union[str, List[str]], **kwargs: Any
    ) -> EmbeddingResponse:
        """Convenience method for generating embeddings.

        Creates an EmbeddingRequest from the input text and additional parameters,
        then delegates to the embed() method for processing.

        Args:
            input_text: The text(s) to embed.
            **kwargs: Additional parameters for the embedding request.

        Returns:
            The embedding response with vector representations.
        """
        request = EmbeddingRequest(input=input_text, **kwargs)
        return self.embed(request=request)