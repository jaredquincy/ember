from __future__ import annotations

from collections import Counter
from typing import List, Optional

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types import EmberModel

from ember.core.utils.eval.evaluators import DiversityEnsembledEvaluator
from ember.core.utils.embedding_utils import Text_Embedding_Ada_002_Model
from ember.core.registry.model.base.services.model_service import ModelService


class DiversityScoringOperatorInputs(EmberModel):
    """Input model for DiversityScoringOperator.

    Attributes:
        responses (List[str]): A list of response strings.
    """

    responses: List[str]
    model_service: ModelService


class DiversityScoringOperatorOutputs(EmberModel):
    """Output model for DiversityScoringOperator.

    Attributes:
        responses (List[str]): A list of response strings.
        diversity score (int): A score representing the diversity between all responses.
        
    """

    responses: List[str]
    diversity_score: int


class DiversityScoringOperator(
    Operator[DiversityScoringOperatorInputs, DiversityScoringOperatorOutputs]
):
    """Operator to aggregate all responses and run a score of a diversity-based metric."""

    specification: Specification = Specification(
        input_model=DiversityScoringOperatorInputs,
        structured_output=DiversityScoringOperatorOutputs,
    )

    def forward(
        self, *, inputs: DiversityScoringOperatorInputs
    ) -> DiversityScoringOperatorOutputs:
        if not inputs.responses or not inputs.model_service:
            return {"responses": None, "diversity_score": 0}
        
        return {"responses": inputs.responses, "divserity_score": DiversityEnsembledEvaluator().evaluate(inputs.responses, embedding_model=Text_Embedding_Ada_002_Model(llm=inputs.model_service))['score']}