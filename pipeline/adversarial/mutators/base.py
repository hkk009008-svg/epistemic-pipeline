"""Abstract base class for modular adversarial mutators."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline.adversarial.models import (
    AdversarialMutationConfig,
    AttackVectorType,
    DifficultyTierEnum,
    DifficultyTierType,
    MultiDomainScenario,
    PoisonedScenarioCase,
)


class BaseAdversarialMutator(ABC):
    """Abstract base class for modular adversarial mutators."""

    @property
    @abstractmethod
    def attack_vector(self) -> AttackVectorType:
        """The specific attack vector handled by this mutator."""
        ...

    @abstractmethod
    def mutate(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum,
        config: AdversarialMutationConfig | None = None,
    ) -> PoisonedScenarioCase:
        """Mutate a clean scenario into an adversarial test case for the given tier."""
        ...
