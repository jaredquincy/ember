from typing import List, Dict, Any
from src.avior.registry.operator.operator_base import Operator, OperatorType, OperatorMetadata
from src.avior.modules.lm_modules import LMModule
from src.avior.registry.prompt_signature.signatures import Signature
from src.avior.registry.prompt_signature.hallucination_signatures import (
    QAHallucinationSignature,
    DialogueHallucinationSignature,
    SummarizationHallucinationSignature,
    HallucinationInputs,
    HallucinationOutputs
)

class HallucinationDetectorOperator(Operator[HallucinationInputs, HallucinationOutputs]):
    """Base operator for hallucination detection. This is a FAN_IN operator that combines
    multiple model judgments into a single output through majority voting."""
    metadata = OperatorMetadata(
        code="HALLUCINATION_DETECTOR",
        description="Base operator for hallucination detection",
        operator_type=OperatorType.FAN_IN,
    )
    
    def __init__(self, lm_modules: List[LMModule], signature: Signature):
        super().__init__(
            name="HallucinationDetector",
            lm_modules=lm_modules,
            signature=signature
        )

    def forward(self, inputs: HallucinationInputs) -> HallucinationOutputs:
        judgements = []
        
        for lm in self.lm_modules:
            # Use signature to format prompt
            prompt = self.build_prompt(inputs.model_dump())
            result = lm(prompt)
            judgements.append(result.strip())

        print("Judgements: ", judgements)
        # Majority vote (FAN_IN operation)
        yes_count = sum(1 for j in judgements if "Yes" in j)
        no_count = sum(1 for j in judgements if "No" in j)
        final_judgement = "Hallucinated" if yes_count > no_count else "Not Hallucinated"

        return HallucinationOutputs(
            judgement=final_judgement,
        )

class QAHallucinationOperator(HallucinationDetectorOperator):
    def __init__(self, lm_modules: List[LMModule]):
        super().__init__(lm_modules, QAHallucinationSignature())

class DialogueHallucinationOperator(HallucinationDetectorOperator):
    def __init__(self, lm_modules: List[LMModule]):
        super().__init__(lm_modules, DialogueHallucinationSignature())

class SummarizationHallucinationOperator(HallucinationDetectorOperator):
    def __init__(self, lm_modules: List[LMModule]):
        super().__init__(lm_modules, SummarizationHallucinationSignature()) 