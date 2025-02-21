from pathlib import Path
from typing import Dict, Any

import yaml

from src.ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider


class MistralDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery provider for Mistral models.
    
    This provider reads model metadata from the mistral_config.yaml file
    and handles discovery of locally available Mistral models.
    """
    
    def __init__(self) -> None:
        """Initialize the Mistral discovery provider."""
        self.config_path = Path(__file__).parent / "mistral_config.yaml"
        
    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve model metadata from the config file.
        
        Returns:
            Dict[str, Dict[str, Any]]: A mapping of model IDs to their metadata.
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            
        models = {}
        for model in config["models"]:
            model_id = f"mistral:{model['id']}"
            models[model_id] = {
                **model,
                "provider": "mistral",
                "requires_model_path": True,
                "requires_api_key": False
            }
                
        return models 