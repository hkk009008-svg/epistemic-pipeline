from pipeline.state import GraphState
from pipeline.models import OpenAIConfig, CurationAnswers
import json
import config
from openai import AsyncOpenAI
from pipeline.helpers import extract_json
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

class CandidateEvaluation(BaseModel):
    name: str = Field(description="Short name of the solution / hypothesis")
    description: str = Field(description="Brief explanation of the solution / hypothesis")
    goal_match: float = Field(description="Raw score 0.0-1.0 estimating goal alignment")
    safety_confidence: float = Field(description="Raw score 0.0-1.0 estimating safety")
    feasibility: float = Field(description="Raw score 0.0-1.0 estimating technical/financial feasibility")
    user_preference: float = Field(description="Raw score 0.0-1.0 estimating user preference match")
    cost_efficiency: float = Field(description="Raw score 0.0-1.0 estimating cost efficiency")
    availability: float = Field(description="Raw score 0.0-1.0 estimating resource availability")
    consensus: float = Field(description="Raw score 0.0-1.0 estimating general industry consensus")

    @field_validator('goal_match', 'safety_confidence', 'feasibility', 'user_preference', 
                     'cost_efficiency', 'availability', 'consensus', mode='after')
    @classmethod
    def clamp_scores(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

class StructuralPayload(BaseModel):
    search_and_audit_log: str = Field(description="Internal log of data search, chunking, and validation")
    problem_solving_log: str = Field(description="Internal log of segment routing and evaluation strategy")
    audited_facts: List[str] = Field(description="List of verified, objective facts")
    identified_gaps: List[str] = Field(description="Actionable unknowns or missing data")
    structural_constraints: List[str] = Field(description="Hard limits, budget, or hardware constraints")
    discriminators: List[str] = Field(description="Factors that differentiate the solutions")
    logical_deductions: List[str] = Field(description="Maximum of 2 hypotheses")
    evidence_boundary: str = Field(description="What is known vs unknown")
    confidence_score: str = Field(description="Overall confidence level (e.g., High, Medium, Low)")
    evaluated_candidates: List[CandidateEvaluation] = Field(description="Exactly 3 to 5 candidates / hypotheses")
    curation_vectors: CurationAnswers = Field(description="MANDATORY: You must extract the user's demographic and health vectors into this exact strict schema for physical product curation.")

    @model_validator(mode='after')
    def auto_heal_constraints(self) -> 'StructuralPayload':
        if len(self.logical_deductions) > 2:
            self.logical_deductions = self.logical_deductions[:2]
        while len(self.evaluated_candidates) < 3:
            self.evaluated_candidates.append(
                CandidateEvaluation(
                    name="Insufficient Data", description="LLM failed to generate alternative.",
                    goal_match=0.0, safety_confidence=0.0, feasibility=0.0, user_preference=0.0, 
                    cost_efficiency=0.0, availability=0.0, consensus=0.0
                )
            )
        return self

class RankedCandidate(BaseModel):
    candidate: CandidateEvaluation
    score: float

class RankedSolutions(BaseModel):
    primary: RankedCandidate
    alternatives: List[RankedCandidate]

class MatrixCalculator:
    WEIGHTS = {
        'goal': 0.50, 'safety': 0.20, 'feas': 0.10, 
        'pref': 0.08, 'cost': 0.06, 'avail': 0.04, 'cons': 0.02
    }

    @classmethod
    def process_and_rank(cls, candidates: List[CandidateEvaluation]) -> RankedSolutions:
        ranked = []
        for c in candidates:
            total_score = (
                (c.goal_match * cls.WEIGHTS['goal']) +
                (c.safety_confidence * cls.WEIGHTS['safety']) +
                (c.feasibility * cls.WEIGHTS['feas']) +
                (c.user_preference * cls.WEIGHTS['pref']) +
                (c.cost_efficiency * cls.WEIGHTS['cost']) +
                (c.availability * cls.WEIGHTS['avail']) +
                (c.consensus * cls.WEIGHTS['cons'])
            )
            ranked.append(RankedCandidate(candidate=c, score=total_score))
        
        ranked.sort(key=lambda x: x.score, reverse=True)
        top_3 = ranked[:3] 

        return RankedSolutions(
            primary=top_3[0],
            alternatives=[top_3[1], top_3[2]]
        )

async def stage_evaluate(state: GraphState) -> dict:
    """Evaluate the user's prompt using the Deterministic Matrix architecture."""
    prompt = state.get("prompt", "")
    mode = state.get("mode", "verify")
    hook_results = state.get("hook_results", [])
    state["emit"]("status", f"Evaluating claim using Deterministic Matrix Engine (Mode: {mode.upper()})...")

    cfg = config.get_stage_config("gpt1")
    client = AsyncOpenAI(api_key=cfg["api_key"])

    # Prepare context
    context = []
    for hr in hook_results:
        context.append(f"--- Tool: {hr.tool} ---\nNote: {hr.note}\nData: {json.dumps(hr.data, indent=2)}")
    
    context_str = "\n".join(context)

    sys_prompt = f"""You are an autonomous Information Auditing and Problem-Solving AI.
### CORE DIRECTIVES
1. Search, Process, and Chunk data systematically.
2. Information Audit: Validate all data against strict epistemic standards using the provided context.
3. Segmented Problem Solving: Route audited chunks to specialized internal logic.
4. Objective Authority: Prioritize factual accuracy. Explicitly acknowledge unknowns.
5. Graceful Safety Handling: Never output a cold system error. Pivot gracefully if needed.

### ENGINE MODE = {mode.upper()}
If mode is VERIFY, your generated candidates should represent conflicting factual hypotheses or interpretations to find the most accurate truth.
If mode is DECISION, your generated candidates should represent distinct alternative paths, solutions, or choices to the user's objective problem.

### DETERMINISTIC LOGIC ENGINE
Do NOT calculate the final matrix score. You will provide exact raw component estimates (0.0 to 1.0) for candidate solutions / hypotheses.
The Python execution environment will calculate the mathematical matrix deterministically.

### VALIDATION RULES
- P1 Abstention: Truthful abstention is better than plausible completion.
- P2 Evidence-Boundedness: Cite verifiable sources. No unsourced causal claims.
- P3 Epistemic Separation: Keep Observed facts, Logical deductions, and Unknowns distinct.

Return ONLY valid JSON that precisely matches the following JSON schema:
{json.dumps(StructuralPayload.model_json_schema(), indent=2)}
"""

    user_prompt = f"Claim to evaluate:\n{prompt}\n\nContext from tools:\n{context_str}"

    try:
        max_retries = 3
        retry_count = 0
        payload = None
        validation_error_msg = ""
        
        while retry_count < max_retries:
            try:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                if retry_count > 0:
                    messages.append({"role": "user", "content": f"CRITICAL: Schema Validation Error in previous attempt. You MUST correct your JSON to match the strict Literals: {validation_error_msg}"})
                    
                resp = await client.chat.completions.create(
                    model=cfg["model"],
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    stream=True
                )
                raw_output = ""
                async for chunk in resp:
                    if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        raw_output += delta
                        state["emit"]("token", delta)

                parsed = extract_json(raw_output)
                
                # Hydrate strongly-typed Pydantic model (applies auto-heal logic)
                payload = StructuralPayload(**parsed)
                break
            except ValidationError as e:
                validation_error_msg = str(e)
                retry_count += 1
                state["emit"]("status", f"V7 Engine Schema Drift Detected. Self-healing loop initiated (Retry {retry_count}/3)...")
            except Exception as e:
                validation_error_msg = str(e)
                retry_count += 1

        if payload is None:
            state["emit"]("status", "V7 Engine Schema Validation Failed 3 times. Activating Graceful Degradation Protocol.")
            # Graceful Degradation Fallback
            payload = StructuralPayload(
                search_and_audit_log="LLM Schema Drift Error. Self-healing failed.",
                problem_solving_log="Triggered Graceful Degradation Fallback.",
                audited_facts=[],
                identified_gaps=["System could not extract safe curation vectors."],
                structural_constraints=[],
                discriminators=[],
                logical_deductions=["Fallback triggered to protect user safety."],
                evidence_boundary="Strict Curation Schema Enforced",
                confidence_score="Low",
                evaluated_candidates=[],
                curation_vectors=CurationAnswers(
                    primary_goal="basic_nutrition",
                    form_factor="any",
                    restrictions=["none"],
                    medication_flag=True, # Forces TS engine to return REVIEW_REQUIRED
                    pregnancy_flag=False,
                    age_band="19_29",
                    budget_band="no_preference"
                )
            )
            
        # Process deterministic calculations natively
        ranking = MatrixCalculator.process_and_rank(payload.evaluated_candidates)
        prim = ranking.primary
            
        md = []
        if mode == "verify":
            # TRUTHLENS: Stripped-down B2C reading experience for fact-checking
            md.append("## Truth Verification Analysis")
            md.append(f"**Confidence:** {payload.confidence_score} | **Boundary:** {payload.evidence_boundary}\n")
            
            md.append("### 🎯 Final Conclusion")
            md.append(f"> **{prim.candidate.name}**")
            md.append(f"> {prim.candidate.description}\n")
            
            if payload.audited_facts:
                md.append("### 📌 Verified Facts")
                for f in payload.audited_facts:
                    md.append(f"- {f}")
            if payload.identified_gaps:
                md.append("\n### ⚠️ Missing Context")
                for g in payload.identified_gaps:
                    md.append(f"- {g}")
                    
        else:
            # OMNIRESOLVE: Full breakdown of matrix and alternate paths for decision making
            md.append(f"## Epistemic Decision Matrix")
            md.append(f"**Confidence Scope:** {payload.confidence_score} | **Evidence Boundary:** {payload.evidence_boundary}\n")
            
            md.append("### 1. What This Means (The Directive)")
            md.append(f"**Primary Path:** {prim.candidate.name}")
            md.append(f"> {prim.candidate.description}\n")
            
            md.append("### 2. How This Was Calculated (Deterministic Matrix)")
            md.append(f"*The engine deterministically weighted these neural assessments to achieve a Final Certainty of **{prim.score:.3f}**:*")
            
            c = prim.candidate
            md.append(f"- **Goal Alignment** *(50% weight)*: {c.goal_match}")
            md.append(f"- **Safety Confidence** *(20% weight)*: {c.safety_confidence}")
            md.append(f"- **Feasibility** *(10% weight)*: {c.feasibility}")
            md.append(f"- **User Preference** *(8% weight)*: {c.user_preference}")
            md.append(f"- **Cost Efficiency** *(6% weight)*: {c.cost_efficiency}")
            md.append(f"- **Availability** *(4% weight)*: {c.availability}")
            md.append(f"- **Consensus** *(2% weight)*: {c.consensus}\n")

            md.append("### 3. Why This Is Correct (The Evidence)")
            if payload.audited_facts:
                md.append("**Verified Facts:**")
                for f in payload.audited_facts:
                    md.append(f"- {f}")
            if payload.logical_deductions:
                md.append("\n**Logical Deductions:**")
                for d in payload.logical_deductions:
                    md.append(f"- {d}")
            if payload.identified_gaps:
                md.append("\n**Identified Gaps (Unknowns):**")
                for g in payload.identified_gaps:
                    md.append(f"- {g}")

            for i, alt in enumerate(ranking.alternatives, 1):
                if "Insufficient" not in alt.candidate.name and "Failed" not in alt.candidate.description:
                    md.append(f"- **{alt.candidate.name}**: {alt.candidate.description} *(Score: {alt.score:.3f})*")
            
        md.append("\n### ⚙️ System Telemetry")
        md.append(f"**Search & Audit Log:**\n{payload.search_and_audit_log}\n")
        md.append(f"**Problem Solving Log:**\n{payload.problem_solving_log}\n")
                 
        final_markdown = "\n".join(md)
        verdict = "PASS" if prim.score > 0.70 and "high" in payload.confidence_score.lower() else "FAIL"

        return {
            "evaluation_result": final_markdown,
            "final_verdict": verdict,
            "reasoning": [f"Primary Verdict: {prim.candidate.name}"],
            "verdict_label": "High Confidence" if verdict == "PASS" else "Needs Review"
        }
    except Exception as e:
        return {
            "evaluation_result": f"System Error executing Matrix Deterministic Logic: {str(e)}",
            "final_verdict": "FAIL",
            "reasoning": [f"Matrix Evaluation failed: {e}"],
            "verdict_label": "Error"
        }

async def stage_re_evaluate(state: GraphState) -> dict:
    """The Recovery Edge triggered by the TS Deterministic Curator."""
    from openai import AsyncOpenAI
    from pipeline.models import CurationAnswers
    from pydantic import ValidationError

    failed_goal = state.get("failed_goal", "unknown")
    prev_context = state.get("previous_cognitive_workspace", "")
    
    state["emit"]("status", f"TS Curator rejected primary path ({failed_goal}). Re-Evaluating Alternative Paths...")

    cfg = config.get_stage_config("gpt1")
    client = AsyncOpenAI(api_key=cfg["api_key"])

    sys_prompt = f"""You are the V7 Epistemic Brain's re-evaluation kernel.
    
### CORE RECOVERY DIRECTIVE
The Deterministic TS Curator rejected your primary path due to 0 inventory matches for the goal: '{failed_goal}'.
Parse your previous cognitive workspace and immediately extract new `CurationAnswers` for your Alternative Path 1.

### VALIDATION RULES
Focus specifically on adjusting `primary_goal`, `form_factor`, or `budget_band` to safely navigate the inventory constraints while adhering to the core epistemic truth. Ensure you map back to the strict JSON required schema for `StructuralPayload`.

Return ONLY valid JSON that matches the StructuralPayload schema.
"""

    user_prompt = f"PREVIOUS COGNITIVE WORKSPACE DATA:\n{prev_context}\n\nFAILURE: INVENTORY_OOM for goal '{failed_goal}'. Action: Pivot to Alternative Path 1."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        max_retries = 3
        retry_count = 0
        payload = None
        validation_error_msg = ""
        
        while retry_count < max_retries:
            if validation_error_msg:
                messages.append({"role": "user", "content": f"Schema Validation Failed. Fix this: {validation_error_msg}"})
                state["emit"]("status", f"Pydantic Validation failed. Triggering Auto-Heal Retry ({retry_count}/{max_retries})...")
            
            resp = await client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                stream=True
            )
            raw_output = ""
            async for chunk in resp:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    raw_output += delta
                    state["emit"]("token", delta)

            parsed = extract_json(raw_output)
            
            try:
                payload = StructuralPayload(**parsed)
                break
            except ValidationError as e:
                validation_error_msg = str(e)
                retry_count += 1

        if payload is None:
            state["emit"]("status", "System HALT: Re-Evaluation Schema Drift Failed 3 Retries. Firing Graceful Degradation Payload.")
            payload = StructuralPayload(
                search_and_audit_log="Auto-Healer Failed.",
                problem_solving_log="Re-evaluation collapsed.",
                audited_facts=[], identified_gaps=[], structural_constraints=[], discriminators=[], logical_deductions=[],
                evidence_boundary="Degradation Protocol Activated",
                confidence_score="SYSTEM OUT OF SYNC",
                evaluated_candidates=[],
                curation_vectors=CurationAnswers(
                    primary_goal="undecided", form_factor="any", restrictions=["none"],
                    medication_flag=True, pregnancy_flag=False, age_band="30_39", budget_band="no_preference"
                )
            )
            raw_output = payload.model_dump_json()

    except Exception as e:
        state["emit"]("error", f"Re-Evaluation Engine Failure: {str(e)}")
        raise e

    state["final_response"] = PipelineResponse(
        prompt="RE-EVALUATION PIVOT",
        hook_results=[],
        final_verdict=payload.confidence_score,
        verdict_label="Alternative Curation Extracted",
        reasoning="",
        final_result=raw_output
    )
    return state
