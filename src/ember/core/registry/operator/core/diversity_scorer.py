from __future__ import annotations

from collections import Counter
from typing import List, Optional

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types import EmberModel

from ember.core.utils.eval.evaluators import DiversityEnsembledEvaluator
from ember.core.registry.model.examples.provider_extension_guide import EmbeddingProviderModel

import logging

class DiversityScoringOperatorInputs(EmberModel):
    """Input model for DiversityScoringOperator.

    Attributes:
        responses (List[str]): A list of response strings.
    """

    responses: List[str]

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
    def __init__(self, *, embedding_model: EmbeddingProviderModel) -> None:
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            logging.warning("DiversityScoringEvaluator isn't initialized with an embedding model")


    def forward(
        self, *, inputs: DiversityScoringOperatorInputs
    ) -> DiversityScoringOperatorOutputs:
        if not inputs.responses or not inputs.model_service:
            return {"responses": None, "diversity_score": 0}
        
        score = DiversityEnsembledEvaluator(embedding_model=self.embedding_model).evaluate(inputs.responses).score
        # logger instead
        logging.info(f"DiversityScoringOperator's score from {len(inputs.responses)} responses: {score}")

        return {"responses": inputs.responses, "diversity_score": score}