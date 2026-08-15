"""Modular Adversarial Mutators Registry and Base Interface.

Provides deterministic mutators across the 6 core attack vectors:
1. StatisticalFallacyMutator
2. PromptInjectionMutator
3. NumericTemporalDriftMutator
4. SyntacticEntanglementMutator
5. CitationDriftMutator
6. BoundarySaturationMutator
"""
from __future__ import annotations

from pipeline.adversarial.models import (
    AttackVectorEnum,
    AttackVectorType,
)
from pipeline.adversarial.mutators.base import BaseAdversarialMutator
from pipeline.adversarial.mutators.boundary_saturation import BoundarySaturationMutator
from pipeline.adversarial.mutators.citation_drift import CitationDriftMutator
from pipeline.adversarial.mutators.entanglement import SyntacticEntanglementMutator
from pipeline.adversarial.mutators.injection import PromptInjectionMutator
from pipeline.adversarial.mutators.numeric_temporal import NumericTemporalDriftMutator
from pipeline.adversarial.mutators.statistical import StatisticalFallacyMutator

MUTATOR_REGISTRY: dict[str, type[BaseAdversarialMutator]] = {
    AttackVectorEnum.STATISTICAL_FALLACY.value: StatisticalFallacyMutator,
    AttackVectorEnum.PROMPT_INJECTION.value: PromptInjectionMutator,
    AttackVectorEnum.NUMERIC_TEMPORAL_DRIFT.value: NumericTemporalDriftMutator,
    AttackVectorEnum.SYNTACTIC_ENTANGLEMENT.value: SyntacticEntanglementMutator,
    AttackVectorEnum.CITATION_DRIFT.value: CitationDriftMutator,
    AttackVectorEnum.POISONING_SATURATION.value: BoundarySaturationMutator,
}


def get_mutator(vector: AttackVectorType | AttackVectorEnum | str) -> BaseAdversarialMutator:
    """Retrieve an instantiated mutator for the specified attack vector.

    Args:
        vector: Attack vector enum value or string.

    Returns:
        Instantiated BaseAdversarialMutator subclass.
    """
    key = vector.value if isinstance(vector, AttackVectorEnum) else str(vector)
    mutator_cls = MUTATOR_REGISTRY.get(key)
    if not mutator_cls:
        raise ValueError(
            f"Unrecognized attack vector '{key}'. Available vectors: {list(MUTATOR_REGISTRY.keys())}"
        )
    return mutator_cls()


__all__ = [
    "MUTATOR_REGISTRY",
    "BaseAdversarialMutator",
    "BoundarySaturationMutator",
    "CitationDriftMutator",
    "NumericTemporalDriftMutator",
    "PromptInjectionMutator",
    "StatisticalFallacyMutator",
    "SyntacticEntanglementMutator",
    "get_mutator",
]
