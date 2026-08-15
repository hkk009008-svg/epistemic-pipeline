"""Adversarial Mutation Engine and Suite Generator.

Provides:
- AdversarialMutationEngine: Core deterministic mutation engine.
- generate_attack_matrix(): Generates the full 24-point (6 vectors x 4 tiers) attack matrix.
- generate_adversarial_suite(): Generates comprehensive poisoned test cases across scenarios.
"""
from __future__ import annotations

from typing import Any

from pipeline.adversarial.models import (
    AdversarialMutationConfig,
    AttackVectorEnum,
    AttackVectorType,
    DifficultyTierEnum,
    DifficultyTierType,
    MultiDomainScenario,
    PoisonedScenarioCase,
    load_scenario_corpus,
)
from pipeline.adversarial.mutators import (
    BaseAdversarialMutator,
    BoundarySaturationMutator,
    CitationDriftMutator,
    NumericTemporalDriftMutator,
    PromptInjectionMutator,
    StatisticalFallacyMutator,
    SyntacticEntanglementMutator,
    get_mutator,
)


class AdversarialMutationEngine:
    """Deterministic generator for multi-vector adversarial stress cases."""

    def __init__(self) -> None:
        self._mutators: dict[str, BaseAdversarialMutator] = {
            AttackVectorEnum.STATISTICAL_FALLACY.value: StatisticalFallacyMutator(),
            AttackVectorEnum.PROMPT_INJECTION.value: PromptInjectionMutator(),
            AttackVectorEnum.NUMERIC_TEMPORAL_DRIFT.value: NumericTemporalDriftMutator(),
            AttackVectorEnum.SYNTACTIC_ENTANGLEMENT.value: SyntacticEntanglementMutator(),
            AttackVectorEnum.CITATION_DRIFT.value: CitationDriftMutator(),
            AttackVectorEnum.POISONING_SATURATION.value: BoundarySaturationMutator(),
        }

    def mutate(
        self,
        scenario: MultiDomainScenario,
        config: AdversarialMutationConfig,
    ) -> PoisonedScenarioCase:
        """Mutate a scenario according to the provided AdversarialMutationConfig.

        Args:
            scenario: The verified ground-truth MultiDomainScenario.
            config: Configuration specifying attack vector, tier, and parameters.

        Returns:
            A deterministic PoisonedScenarioCase with calculated defense expectations.
        """
        vec_key = config.attack_vector.value if isinstance(config.attack_vector, AttackVectorEnum) else str(config.attack_vector)
        mutator = self._mutators.get(vec_key) or get_mutator(vec_key)
        return mutator.mutate(scenario=scenario, tier=config.difficulty_tier, config=config)

    def mutate_scenario(
        self,
        scenario: MultiDomainScenario,
        vector: AttackVectorType | AttackVectorEnum | str,
        tier: DifficultyTierType | DifficultyTierEnum | str,
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Convenience method to mutate a scenario with vector and tier.

        Args:
            scenario: Target MultiDomainScenario.
            vector: Attack vector type or enum.
            tier: Difficulty tier type or enum.
            **kwargs: Additional mutation parameters.

        Returns:
            PoisonedScenarioCase.
        """
        vec_val = vector.value if isinstance(vector, AttackVectorEnum) else str(vector)
        tier_val = tier.value if isinstance(tier, DifficultyTierEnum) else str(tier)
        config = AdversarialMutationConfig(
            attack_vector=vec_val,  # type: ignore
            difficulty_tier=tier_val,  # type: ignore
            mutation_parameters=kwargs,
        )
        return self.mutate(scenario, config)

    def mutate_statistical(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum | str = "moderate",
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Generate statistical fallacy attack."""
        return self.mutate_scenario(scenario, AttackVectorEnum.STATISTICAL_FALLACY, tier, **kwargs)

    def mutate_injection(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum | str = "moderate",
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Generate prompt injection attack."""
        return self.mutate_scenario(scenario, AttackVectorEnum.PROMPT_INJECTION, tier, **kwargs)

    def mutate_numeric_temporal(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum | str = "moderate",
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Generate numeric/temporal drift attack."""
        return self.mutate_scenario(scenario, AttackVectorEnum.NUMERIC_TEMPORAL_DRIFT, tier, **kwargs)

    def mutate_entanglement(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum | str = "moderate",
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Generate syntactic entanglement attack."""
        return self.mutate_scenario(scenario, AttackVectorEnum.SYNTACTIC_ENTANGLEMENT, tier, **kwargs)

    def mutate_citation_drift(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum | str = "moderate",
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Generate citation drift attack."""
        return self.mutate_scenario(scenario, AttackVectorEnum.CITATION_DRIFT, tier, **kwargs)

    def mutate_boundary_saturation(
        self,
        scenario: MultiDomainScenario,
        tier: DifficultyTierType | DifficultyTierEnum | str = "moderate",
        **kwargs: Any,
    ) -> PoisonedScenarioCase:
        """Generate boundary saturation attack."""
        return self.mutate_scenario(scenario, AttackVectorEnum.POISONING_SATURATION, tier, **kwargs)


def generate_attack_matrix() -> list[AdversarialMutationConfig]:
    """Generate the complete 24-point attack matrix (6 vectors x 4 tiers).

    Returns:
        List of 24 AdversarialMutationConfig instances.
    """
    matrix: list[AdversarialMutationConfig] = []
    vectors = [
        AttackVectorEnum.STATISTICAL_FALLACY,
        AttackVectorEnum.PROMPT_INJECTION,
        AttackVectorEnum.NUMERIC_TEMPORAL_DRIFT,
        AttackVectorEnum.SYNTACTIC_ENTANGLEMENT,
        AttackVectorEnum.CITATION_DRIFT,
        AttackVectorEnum.POISONING_SATURATION,
    ]
    tiers = [
        DifficultyTierEnum.MILD,
        DifficultyTierEnum.MODERATE,
        DifficultyTierEnum.EXTREME,
        DifficultyTierEnum.BREAKING,
    ]

    tier_unbacked_defaults = {
        DifficultyTierEnum.MILD: 0.15,
        DifficultyTierEnum.MODERATE: 0.30,
        DifficultyTierEnum.EXTREME: 0.40,
        DifficultyTierEnum.BREAKING: 0.80,
    }

    tier_hard_violations_defaults = {
        DifficultyTierEnum.MILD: 1,
        DifficultyTierEnum.MODERATE: 1,
        DifficultyTierEnum.EXTREME: 2,
        DifficultyTierEnum.BREAKING: 3,
    }

    for vec in vectors:
        for t in tiers:
            matrix.append(
                AdversarialMutationConfig(
                    attack_vector=vec.value,  # type: ignore
                    difficulty_tier=t.value,  # type: ignore
                    target_unbacked_ratio=tier_unbacked_defaults[t],
                    target_hard_violations=tier_hard_violations_defaults[t],
                    mutation_parameters={"syntactic_depth": 1 if t == DifficultyTierEnum.MILD else 3},
                )
            )
    return matrix


def generate_adversarial_suite(
    scenarios: list[MultiDomainScenario] | None = None,
    attack_matrix: list[AdversarialMutationConfig] | None = None,
) -> list[PoisonedScenarioCase]:
    """Generate a comprehensive adversarial test suite across scenarios and configurations.

    Args:
        scenarios: List of MultiDomainScenario objects (defaults to loading full corpus).
        attack_matrix: List of AdversarialMutationConfig objects (defaults to 24-point matrix).

    Returns:
        List of PoisonedScenarioCase test cases.
    """
    target_scenarios = scenarios if scenarios is not None else load_scenario_corpus()
    configs = attack_matrix if attack_matrix is not None else generate_attack_matrix()
    engine = AdversarialMutationEngine()

    cases: list[PoisonedScenarioCase] = []
    for sc in target_scenarios:
        for cfg in configs:
            case = engine.mutate(scenario=sc, config=cfg)
            cases.append(case)

    return cases
