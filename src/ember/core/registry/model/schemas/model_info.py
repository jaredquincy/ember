from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo

from ember.core.registry.model.core.schemas.cost import ModelCost, RateLimit
from ember.core.registry.model.core.schemas.provider_info import ProviderInfo


class ModelInfo(BaseModel):
    """Metadata and configuration for instantiating a model.

    Attributes:
        model_id (str): Unique identifier for the model.
        model_name (str): Human-readable name of the model.
        cost (ModelCost): Cost details associated with the model.
        rate_limit (RateLimit): Rate limiting parameters for model usage.
        provider (ProviderInfo): Provider information containing defaults and endpoints.
        api_key (Optional[str]): API key for authentication. If omitted, the provider's default API key is used.
    """

    model_id: str = Field(...)
    model_name: str = Field(...)
    cost: ModelCost
    rate_limit: RateLimit
    provider: ProviderInfo
    api_key: Optional[str] = None

    @field_validator("api_key", mode="before")
    def validate_api_key(cls, api_key: Optional[str], info: ValidationInfo) -> str:
        """Ensures an API key is provided, either explicitly or via the provider.

        This validator checks if an API key is supplied. If not, it attempts to obtain a default
        API key from the associated provider. A ValueError is raised if neither is available.

        Args:
            api_key (Optional[str]): The API key provided before validation.
            info (ValidationInfo): Validation context containing additional field data.

        Returns:
            str: A valid API key.

        Raises:
            ValueError: If no API key is provided and the provider lacks a default.
        """
        provider_obj = info.data.get("provider")
        if not api_key and (not provider_obj or not provider_obj.default_api_key):
            raise ValueError("No API key provided or defaulted.")
        return api_key or provider_obj.default_api_key

    def get_api_key(self) -> str:
        """Retrieves the validated API key.

        Returns:
            str: The API key to be used for authentication.
        """
        # Assert that the api_key is set following validation.
        assert (
            self.api_key is not None
        ), "The API key must have been set by the validator."
        return self.api_key

    def get_base_url(self) -> Optional[str]:
        """Retrieves the base URL from the provider, if it exists.

        Returns:
            Optional[str]: The base URL specified by the provider, or None if not available.
        """
        return self.provider.base_url

    model_config = {
        "protected_namespaces": (),  # Disable Pydantic's protected namespace checks.
    }
