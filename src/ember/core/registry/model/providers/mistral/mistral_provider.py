from typing import Any, Dict, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ember.core.registry.model.providers.base_provider import BaseProviderModel
from src.ember.core.registry.model.schemas.model_info import ModelInfo
from src.ember.core.registry.model.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
)


class MistralProvider(BaseProviderModel):
    """Provider for Mistral models using HuggingFace Transformers.
    
    This provider supports running Mistral models locally using the transformers
    library. It handles loading quantized models and efficient inference.
    """
    
    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize the Mistral provider.
        
        Args:
            model_info (ModelInfo): Model metadata including path and configuration
        """
        super().__init__(model_info)
        self.model_path = self._resolve_model_path()
        self.device = "cuda" if self.model_info.use_gpu else "cpu"
        
    def _resolve_model_path(self) -> Path:
        """Resolve the path to the model weights.
        
        Returns:
            Path: Path to the model weights directory
            
        Raises:
            ValueError: If model path is not specified or doesn't exist
        """
        if not self.model_info.model_path:
            raise ValueError(f"Model path must be specified for {self.model_info.model_id}")
            
        path = Path(self.model_info.model_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Model path {path} does not exist")
            
        return path
        
    def create_client(self) -> Dict[str, Any]:
        """Create Mistral model and tokenizer instances.
        
        Returns:
            Dict[str, Any]: Dictionary containing model and tokenizer
        """
        # Load model with 4-bit quantization if on CPU to reduce memory usage
        quantization_config = None
        if self.device == "cpu":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        return {
            "model": model,
            "tokenizer": tokenizer
        }
        
    def forward(self, request: ChatRequest) -> ChatResponse:
        """Run inference using the Mistral model.
        
        Args:
            request (ChatRequest): Chat request containing prompt and parameters
            
        Returns:
            ChatResponse: The model's response
        """
        # Format prompt for chat
        prompt = self._format_prompt(request.prompt)
        
        # Tokenize input
        inputs = self.client["tokenizer"](
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_info.context_length or 8192
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.client["model"].generate(
                **inputs,
                max_new_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.client["tokenizer"].eos_token_id
            )
            
        # Decode response
        response_text = self.client["tokenizer"].decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Calculate token usage
        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        
        return ChatResponse(
            text=response_text.strip(),
            model_id=self.model_info.model_id,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for Mistral chat models.
        
        Args:
            prompt (str): Raw user prompt
            
        Returns:
            str: Formatted prompt with chat markers
        """
        # Mistral chat format
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" 