from typing import Any, Dict, Optional
from pathlib import Path
import llama_cpp

from src.ember.core.registry.model.providers.base_provider import BaseProviderModel
from src.ember.core.registry.model.schemas.model_info import ModelInfo
from src.ember.core.registry.model.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)


class LlamaProvider(BaseProviderModel):
    """Provider for Llama models using llama.cpp.
    
    This provider supports running Llama models locally using the efficient
    llama.cpp implementation. It handles both chat and completion tasks.
    """
    
    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize the Llama provider.
        
        Args:
            model_info (ModelInfo): Model metadata including path and configuration
        """
        super().__init__(model_info)
        self.model_path = self._resolve_model_path()
        self.device = "cuda" if self.model_info.use_gpu else "cpu"
        
    def _resolve_model_path(self) -> Path:
        """Resolve the path to the model weights.
        
        Returns:
            Path: Path to the model weights file
            
        Raises:
            ValueError: If model path is not specified or doesn't exist
        """
        if not self.model_info.model_path:
            raise ValueError(f"Model path must be specified for {self.model_info.model_id}")
            
        path = Path(self.model_info.model_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Model path {path} does not exist")
            
        return path
        
    def create_client(self) -> Any:
        """Create a Llama model instance using llama.cpp.
        
        Returns:
            llama_cpp.Llama: Initialized Llama model
        """
        return llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.model_info.context_length or 4096,
            n_threads=self.model_info.num_threads or 4,
            n_gpu_layers=0 if self.device == "cpu" else -1  # -1 means all layers on GPU
        )
        
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Run inference using the Llama model.
        
        Args:
            request (ChatRequest): Chat request containing prompt and parameters
            
        Returns:
            ChatResponse: The model's response
        """
        # Format prompt for chat
        prompt = self._format_prompt(request.prompt)
        
        # Run inference
        output = self.client.create_completion(
            prompt=prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=0.95,
            stream=False
        )
        
        return ChatResponse(
            text=output["choices"][0]["text"].strip(),
            model_id=self.model_info.model_id,
            usage={
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
                "total_tokens": output["usage"]["total_tokens"]
            }
        )
        
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for Llama chat models.
        
        Args:
            prompt (str): Raw user prompt
            
        Returns:
            str: Formatted prompt with chat markers
        """
        # Basic chat formatting for Llama 2
        return f"### Human: {prompt}\n\n### Assistant:" 