"""Pydantic models for the bare-bones Auditor pipeline."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any


class OpenAIConfig(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini"

class TavilyConfig(BaseModel):
    api_key: str
    enabled: bool = True

class StageConfig(BaseModel):
    stage: str
    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""

class PipelineRequest(BaseModel):
    prompt: str = Field(..., max_length=10_000)
    mode: str = "verify"
    stream: bool = False

class ReEvaluateRequest(BaseModel):
    """Payload for triggering the self-healing bidirectional loop upon TS Engine failure."""
    error: str = Field(default="INVENTORY_OOM")
    failed_goal: str = Field(description="The primary_goal that yielded 0 matches")
    previous_workspace_context: str = Field(description="The raw LLM string to parse and recover from")
    stream: bool = True

class HookResult(BaseModel):
    """Result of a pipeline tool hook execution."""
    tool: str
    data: Any
    note: str = ""

class PipelineResponse(BaseModel):
    """The final response from the bare-bones Auditor pipeline."""
    prompt: str
    hook_results: List[HookResult] = []
    final_verdict: str  # e.g., "PASS", "FAIL"
    verdict_label: str = "" # Human-readable label
    reasoning: str = "" # Final evaluation reasoning
    final_result: str = "" # Final compiled markdown or text response


PrimaryGoalType = Literal["memory_focus", "blood_flow", "immunity_vitality", "gut_stomach", "basic_nutrition", "undecided"]
FormFactorType = Literal["capsule", "tablet", "stick", "liquid", "any"]
RestrictionType = Literal["plant_based_focus", "avoid_animal_ingredients", "check_specific_ingredients", "none"]
AgeBandType = Literal["under_19", "19_29", "30_39", "40_49", "50_plus"]
BudgetBandType = Literal["under_20000", "20000_30000", "30000_50000", "50000_plus", "no_preference"]

class CurationAnswers(BaseModel):
    """Rigid structural handshake connecting Python Epistemic Matrix to TypeScript Deterministic Curation Engine."""
    primary_goal: PrimaryGoalType = Field(description="The primary health focus. Defaults to 'undecided'.")
    form_factor: FormFactorType = Field(description="The preferred physical format of the product. Defaults to 'any'.")
    restrictions: List[RestrictionType] = Field(description="Dietary restrictions. If no known restrictions, pass ['none'].")
    medication_flag: bool = Field(description="CRITICAL: Set to true if query implies medication interaction risks.")
    pregnancy_flag: bool = Field(description="CRITICAL: Set to true if query implies pregnancy or nursing.")
    age_band: AgeBandType = Field(description="The user's presumed age band based on context.")
    budget_band: BudgetBandType = Field(description="The user's budget capacity.")
