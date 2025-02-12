from typing import List, Dict, Any
from pydantic import BaseModel
from src.avior.registry.prompt_signature.signatures import Signature

class HallucinationInputs(BaseModel):
    query: str
    choices: Dict[str, str]

class HallucinationOutputs(BaseModel):
    judgement: str

class QAHallucinationSignature(Signature):
    def __init__(self):
        super().__init__(
            required_inputs=["query", "choices"],
            input_model=HallucinationInputs,
            output_model=HallucinationOutputs,
            prompt_template="""You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on world knowledge. The answer you provide MUST be in the provided choices. Only use the letter of the choice you choose.

#Question#: {query}
#Choices#: {choices}
#Your Judgement#:"""
        )

class DialogueHallucinationSignature(Signature):
    def __init__(self):
        super().__init__(
            required_inputs=["dialogue_history", "response"],
            prompt_template="""You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be "Yes" or "No".

#Dialogue History#: {dialogue_history}
#Response#: {response}
#Your Judgement#:"""
        )

class SummarizationHallucinationSignature(Signature):
    def __init__(self):
        super().__init__(
            required_inputs=["document", "summary"],
            prompt_template="""You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be "Yes" or "No".

#Document#: {document}
#Summary#: {summary}
#Your Judgement#:"""
        )