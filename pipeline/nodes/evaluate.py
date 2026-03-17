from pipeline.state import GraphState
from pipeline.models import OpenAIConfig
import json
import config
from openai import AsyncOpenAI
from pipeline.helpers import extract_json

async def stage_evaluate(state: GraphState) -> dict:
    """Evaluate the user's prompt against the hook results."""
    prompt = state.get("prompt", "")
    hook_results = state.get("hook_results", [])
    state["emit"]("status", "Evaluating claim against tool results...")

    cfg = config.get_stage_config("gpt1")
    client = AsyncOpenAI(api_key=cfg["api_key"])

    # Prepare context
    context = []
    for hr in hook_results:
        context.append(f"--- Tool: {hr.tool} ---\nNote: {hr.note}\nData: {json.dumps(hr.data, indent=2)}")
    
    context_str = "\n".join(context)

    sys_prompt = f"""You are an Auditor evaluator. Evaluate the following subject/claim against the provided tool context.
Return ONLY valid JSON with no markdown in the following schema:
{{
  "verdict": "PASS" | "FAIL",
  "reasoning": ["Step 1...", "Step 2..."]
}}
"""

    user_prompt = f"Claim to evaluate:\n{prompt}\n\nContext from tools:\n{context_str}"

    try:
        resp = await client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        raw_output = resp.choices[0].message.content
        parsed = extract_json(raw_output)
        
        verdict = parsed.get("verdict", "FAIL")
        reasoning = parsed.get("reasoning", ["Failed to parse reasoning."])
        
        return {
            "evaluation_result": raw_output,
            "final_verdict": verdict,
            "reasoning": reasoning,
            "verdict_label": "Verified" if verdict == "PASS" else "Needs Attention"
        }
    except Exception as e:
        return {
            "evaluation_result": str(e),
            "final_verdict": "FAIL",
            "reasoning": [f"Evaluation failed: {e}"],
            "verdict_label": "Error"
        }
