from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import re
import statistics
import time
import uuid
from collections import Counter, defaultdict
import openai
import httpx

app = FastAPI(title="GPT-1 > GPT-2 > GPT-3 Verification Pipeline")


def _require_auth(request: Request) -> None:
    """Optional bearer-token auth for sensitive API routes."""
    required = os.getenv("PIPELINE_AUTH_TOKEN", "").strip()
    if not required:
        return

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    provided = auth.split(" ", 1)[1].strip()
    if provided != required:
        raise HTTPException(status_code=401, detail="Invalid bearer token.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())[:8]
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_id": error_id},
    )


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


# =====================================================
# Models
# =====================================================

class OpenAIConfig(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini"

class ClaimEntry(BaseModel):
    claim: str
    category: str
    justification: str

class EditEntry(BaseModel):
    action: str
    target: str
    replacement: str

class PipelineRequest(BaseModel):
    prompt: str
    gpt1_system: str = ""
    gpt2_system: str = ""
    gpt3_system: str = ""

class PipelineResponse(BaseModel):
    gpt1_input: str
    gpt1_output: str
    bypassed: bool
    # GPT-2 results
    gpt2_raw: str
    claim_table: List[ClaimEntry]
    violations: List[str]
    gpt2_verdict: str
    # GPT-3 arbiter results (only if GPT-2 FAIL)
    arbiter_invoked: bool
    arbiter_decision: str
    arbiter_rationale: List[str]
    arbiter_edits: List[EditEntry]
    arbiter_policy_notes: List[str]
    arbiter_raw: str
    # Rewrite loop (if ALLOW_WITH_EDITS)
    rewrite_occurred: bool
    rewrite_output: str
    rewrite_gpt2_raw: str
    rewrite_claim_table: List[ClaimEntry]
    rewrite_violations: List[str]
    rewrite_verdict: str
    # Final
    final_verdict: str
    final_result: str
    # Prompt routing / sanitizer metadata
    prompt_flags: Optional[dict] = None
    sanitizer_applied: bool = False

_openai_config: dict = {}

MAX_REWRITE_LOOPS = 1  # prevent infinite loops

# Activation phrases that bypass GPT-2 verification
ACTIVATION_PATTERNS = [
    r"active\.$",
    r"Production active\.",
    r"Audit v\d+",
    r"^System initialized",
    r"^Ready\.$",
]

# =====================================================
# System Prompts
# =====================================================

DEFAULT_GPT1_SYSTEM = (
    "You are GPT-1, a structured reasoning and synthesis engine.\n\n"
    "Hard constraints:\n"
    "- No fabricated sources, statutes, studies, metrics, or percentages.\n"
    "- Do not use \"studies/research/data suggest\" unless you provide a specific citation AND a concrete number/quote.\n"
    "- Do not provide advice/options unless the user explicitly asks what to do.\n"
    "- If asked for percentages and none are available in provided/cited evidence, output Unknown(Actionable).\n"
    "- When mentioning professionals (attorneys, brokers, consultants), use ONLY role-definition + uncertainty language.\n"
    "  NEVER use benefit-language (\"could help\", \"could assist\", \"may improve\", \"could potentially\", \"may provide guidance\").\n"
    "  CORRECT: \"An attorney's function is to advise on requirements and prepare/submit filings; whether that changes outcomes is unknown.\"\n"
    "  WRONG: \"An attorney could potentially assist in navigating the process.\"\n\n"
    "Default format:\n"
    "1) Problem Framing\n"
    "2) Assumptions (explicit)\n"
    "3) Analysis (Facts; then Inferences labeled)\n"
    "4) Unknowns (Actionable / Structural)\n"
    "5) Confidence (High/Medium/Low + 1 sentence)\n\n"
    "Only include \"Options\" if user asked for actions/choices."
)

DEFAULT_GPT2_SYSTEM = (
    'You are GPT-2, a strict claim validator.\n\n'
    'Output VALID JSON ONLY (no markdown, no prose, no code fences).\n\n'
    'You will receive both the ORIGINAL PROMPT (what the user asked) and the GPT-1 RESPONSE TO VERIFY.\n'
    'You MUST read the ORIGINAL PROMPT to determine whether the user requested advice/actions.\n\n'
    'Schema:\n'
    '{\n'
    '  "claim_table": [{"claim": "...", "category": "Supported|User-provided|Inference|Hypothesis|Unsupported", "justification": "..."}],\n'
    '  "findings": [{"type": "...", "severity": "hard|soft", "detail": "..."}],\n'
    '  "verdict": "PASS|FAIL"\n'
    '}\n\n'
    'Finding types and severities:\n'
    '  HARD severity (always serious):\n'
    '    - "Fabricated statistic" — invented percentage or number with no source\n'
    '    - "Fabricated citation" — made-up study, paper, or named source\n'
    '    - "False legal conclusion" — claims legality/illegality without verified authority\n'
    '  SOFT severity (minor unless accumulated):\n'
    '    - "Unsupported evidence reference" — vague evidence language without citation\n'
    '    - "Prescriptive creep" — unsolicited advice or outcome promises\n'
    '    - "Overconfidence" — Medium/High confidence with core Unsupported claims\n'
    '    - "Missing jurisdiction" — legal/regulatory claim with ambiguous jurisdiction\n\n'
    'Rules:\n'
    '- "studies/data/research/generally/often/suggests" without citation+numeric/quotable support -> finding type "Unsupported evidence reference", severity "soft"\n'
    '- Prescriptive creep rule (MUST check user prompt):\n'
    '  * If GPT-1 gives advice/options AND the ORIGINAL PROMPT does NOT ask for advice/actions/recommendations -> finding "Prescriptive creep", severity "soft"\n'
    '  * If the ORIGINAL PROMPT explicitly asks about hiring professionals, attorneys, brokers, or asks "should I" / "would it help" / "what should I do":\n'
    '    -> Allow process-only role-definition language (e.g., "an attorney advises on X and prepares filings")\n'
    '    -> Still flag as "Prescriptive creep" ONLY if GPT-1 promises outcomes (e.g., "will improve your chances", "could help you succeed")\n'
    '  * Pure role-definition + uncertainty framing (e.g., "an attorney handles filings; whether that changes outcomes is unknown") is NOT prescriptive creep\n'
    '- Medium/High confidence with core Unsupported -> finding "Overconfidence", severity "soft"\n'
    '- Legal/regulatory with ambiguous jurisdiction -> finding "Missing jurisdiction", severity "soft"\n'
    '- Invented percentage or specific number without any source -> finding "Fabricated statistic", severity "hard"\n'
    '- Made-up study, paper, or source name -> finding "Fabricated citation", severity "hard"\n'
    '- Claims something is legal/illegal without verified statute/authority -> finding "False legal conclusion", severity "hard"\n\n'
    'Verdict rule: FAIL only if any finding has severity "hard", OR if count of "soft" findings >= 3. Otherwise PASS.'
)

DEFAULT_GPT3_SYSTEM = (
    'You are GPT-3, the Arbiter.\n\n'
    'You do NOT answer the user\'s question. You do NOT add any new facts.\n'
    'You only adjudicate whether GPT-2\'s FAIL is:\n'
    '(A) Correct and should block output,\n'
    '(B) Over-strict and can be allowed with constraints,\n'
    '(C) Correct but fixable via minimal edits to GPT-1 output.\n\n'
    'Inputs you will receive:\n'
    '- user_prompt\n'
    '- gpt1_output\n'
    '- gpt2_result_json (contains claim_table, violations, verdict)\n\n'
    'You must output VALID JSON ONLY with this schema:\n'
    '{\n'
    '  "arbiter_decision": "BLOCK" | "ALLOW_WITH_EDITS" | "ALLOW_AS_UNKNOWN_ONLY",\n'
    '  "rationale": ["..."],\n'
    '  "edits_for_gpt1": [\n'
    '    {"action": "DELETE"|"REWRITE"|"MOVE_TO_UNKNOWN", "target": "quoted text fragment from gpt1", "replacement": "..."}\n'
    '  ],\n'
    '  "final_policy_notes": ["..."]\n'
    '}\n\n'
    'Adjudication policy (apply deterministically):\n\n'
    '1) If GPT-1 contains any fabricated statistic, specific legal conclusion, or claim of current legality without verification:\n'
    '   -> arbiter_decision = "BLOCK" unless it can be deleted without harming coherence.\n\n'
    '2) If GPT-2 flags a claim that is a non-quantitative procedural generality (e.g., "cases are assessed individually") '
    'AND GPT-1 already frames it as conditional/uncertain:\n'
    '   -> do NOT block; prefer "ALLOW_WITH_EDITS" to rephrase as inference and/or move to Unknown.\n\n'
    '3) If the user requests a percentage but GPT-1 states that no dataset is available and does NOT invent a number:\n'
    '   -> do NOT block solely for missing percentage. Allow if framed as Unknown(Actionable).\n\n'
    '4) If advice is explicitly requested by the user:\n'
    '   -> allow only if GPT-1 expresses it as conditional process guidance (not promised outcomes).\n'
    '   If GPT-1 implies outcome improvement ("improve your odds") without evidence:\n'
    '   -> require edit: rewrite to "may help with procedure/filings; no guarantee".\n\n'
    '5) If the question is inherently indeterminate ("medically safer" across countries):\n'
    '   -> require "ALLOW_AS_UNKNOWN_ONLY" unless GPT-1 cleanly separates Unknown(Structural) and avoids conclusions.\n\n'
    'Never request web browsing or external actions in the arbiter output.\n'
    'Never introduce new facts or citations.'
)


# =====================================================
# Helpers
# =====================================================

def is_activation_phrase(text: str) -> bool:
    """Check if GPT-1 output is an activation/init phrase that should bypass GPT-2."""
    stripped = text.strip()
    if len(stripped) < 100:
        for pattern in ACTIVATION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                return True
    return False


# ---- Deterministic Prompt Router ----
_ADVICE_RE = re.compile(
    r"(?i)\b(?:what should I do|should I|recommend|steps|best way|would it help"
    r"|what are my options|how do I|how can I|tips for|help me)\b"
)
_PERCENT_RE = re.compile(
    r"(?i)\b(?:percent|percentage|rate|odds|how many|how often"
    r"|probability|fraction|proportion|typically)\b"
)
_LEGAL_RE = re.compile(
    r"(?i)\b(?:legal|illegal|law|regulation|import|export|IRS|SEC"
    r"|compliance|statute|ordinance|ban|prohibited)\b"
)
_JURISDICTION_RE = re.compile(
    r"(?i)\b(?:US|UK|EU|federal|state of"
    r"|United States|United Kingdom|Canada|Australia|Germany|France|India"
    r"|China|Japan|Brazil|Mexico|California|Texas|New York|Florida"
    r"|Ohio|Illinois|Pennsylvania|Georgia|Michigan|Virginia"
    r"|North Carolina|South Carolina|Massachusetts|Washington"
    r"|Arizona|Colorado|Oregon|Nevada|Tennessee|Kentucky"
    r"|Alabama|Louisiana|Maryland|Minnesota|Wisconsin|Missouri"
    r"|Connecticut|Iowa|Arkansas|Mississippi|Kansas|Utah"
    r"|Nebraska|Oklahoma|New Mexico|Hawaii|Idaho|Montana"
    r"|Wyoming|Vermont|Maine|New Hampshire|Rhode Island"
    r"|South Dakota|North Dakota|Delaware|West Virginia|Alaska)\b"
)
_FUTURE_YEAR_RE = re.compile(r"\b(20[3-9]\d|2[1-9]\d{2}|[3-9]\d{3})\b")


def route_prompt(prompt: str) -> dict:
    """Deterministic heuristic router — classifies prompt features for downstream use."""
    return {
        "advice_requested": bool(_ADVICE_RE.search(prompt)),
        "percent_requested": bool(_PERCENT_RE.search(prompt)),
        "legal_mode": bool(_LEGAL_RE.search(prompt)),
        "jurisdiction_present": bool(_JURISDICTION_RE.search(prompt)),
        "future_year": bool(_FUTURE_YEAR_RE.search(prompt)),
    }


# ---- Deterministic Sanitizer ----
_BANNED_EVIDENCE_RE = re.compile(
    r"(?i)\b(?:studies suggest|research shows|data indicates"
    r"|generally|often|typically|commonly|usually)\b"
    r"(?!\s*\([^)]+\))"  # not followed by a parenthetical citation
    r"(?!\s*\[[^\]]+\])"  # not followed by a bracket citation
)
_BARE_PERCENT_RE = re.compile(
    r"(?i)\b(?:about|roughly|approximately|around|nearly|close to|an estimated|estimated)?\s*"
    r"\d+(?:\.\d+)?\s*(?:%|percent)\b"
)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially)\b"
)


def sanitize_output(text: str, flags: dict) -> str:
    """Pre-clean GPT-1 output deterministically before GPT-2 verification."""
    result = text
    # 1. Remove banned evidence phrases (not followed by citation)
    result = _BANNED_EVIDENCE_RE.sub("", result)
    # 2. Convert bare % claims
    result = _BARE_PERCENT_RE.sub(
        "Unknown (Actionable): No authoritative dataset available for this figure", result
    )
    # 3. Strip outcome-promise phrases
    result = _OUTCOME_PROMISE_RE.sub("", result)
    # Clean up residual double-spaces / trailing whitespace per line
    result = re.sub(r"  +", " ", result)
    result = re.sub(r" +\n", "\n", result)
    return result.strip()


def extract_json(raw: str) -> dict:
    """Hardened JSON extractor. Handles fences, prose wrapping, truncation."""
    cleaned = raw.strip()

    # Strip markdown code fences
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break

    # If it starts with prose before JSON, extract the JSON object
    if not cleaned.startswith("{"):
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            cleaned = match.group(0)

    # Try parsing as-is
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fixing truncated JSON by closing brackets
    for suffix in ["}", "]}", '"]}', '"}]}', '"]}]}']:
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from: {raw[:300]}")


def call_openai(client, model: str, system: str, user_content: str, expect_json: bool = False) -> str:
    """Centralized OpenAI call with error handling.

    When *expect_json* is True and the response is not valid JSON, the
    function retries once with a repair instruction asking for valid JSON.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        result = resp.choices[0].message.content

        # Schema-enforced JSON retry
        if expect_json:
            try:
                extract_json(result)
            except (ValueError, json.JSONDecodeError):
                retry_resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": result},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON. "
                                "Please output ONLY valid JSON matching the required schema."
                            ),
                        },
                    ],
                )
                result = retry_resp.choices[0].message.content

        return result
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key. Please re-enter your key.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit hit. Wait a moment and try again.")
    except openai.APITimeoutError:
        raise HTTPException(status_code=504, detail="OpenAI request timed out. Please retry.")
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="OpenAI connection failed. Please retry.")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Upstream timeout contacting model provider.")
    except httpx.NetworkError:
        raise HTTPException(status_code=503, detail="Network error contacting model provider.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI: {str(e)}")


def parse_gpt2(raw: str, flags: Optional[dict] = None):
    """Parse GPT-2 JSON output into claim_table, findings, violations, verdict.

    When *flags* is provided and ``advice_requested`` is True, soft
    "Prescriptive creep" findings that do NOT contain outcome-promise
    language are filtered out before the verdict is recalculated.
    """
    try:
        parsed = extract_json(raw)
        claim_table = [
            ClaimEntry(
                claim=c.get("claim", ""),
                category=c.get("category", "Unknown"),
                justification=c.get("justification", ""),
            )
            for c in parsed.get("claim_table", [])
        ]

        # ---------- findings (new schema) ----------
        raw_findings = parsed.get("findings", [])

        # Backward compat: if GPT-2 returned old "violations" list instead
        if not raw_findings and parsed.get("violations"):
            raw_findings = [
                {"type": v, "severity": "soft", "detail": v}
                for v in parsed["violations"]
            ]

        findings = []  # type: List[dict]
        _OUTCOME_KW = re.compile(
            r"(?i)\b(?:will improve|will reduce|will increase|improve your|"
            r"could help|could assist|may improve|may help|could potentially|"
            r"guarantee|ensure|succeed)\b"
        )
        for f in raw_findings:
            ftype = f.get("type", "")
            severity = f.get("severity", "soft").lower()
            detail = f.get("detail", "")

            # Context-aware filter: if advice was requested, drop soft
            # "Prescriptive creep" unless it contains outcome promises.
            if (
                flags
                and flags.get("advice_requested")
                and ftype == "Prescriptive creep"
                and severity == "soft"
                and not _OUTCOME_KW.search(detail)
            ):
                continue

            findings.append({"type": ftype, "severity": severity, "detail": detail})

        # Derive violations list (backward compat)
        violations = [f["type"] for f in findings]

        # Recompute verdict based on severity-tier rule
        hard_count = sum(1 for f in findings if f["severity"] == "hard")
        soft_count = sum(1 for f in findings if f["severity"] == "soft")
        if hard_count > 0 or soft_count >= 3:
            verdict = "FAIL"
        else:
            verdict = "PASS"

        return claim_table, violations, verdict, findings
    except Exception:
        return (
            [],
            ["GPT-2 parse error: could not extract valid JSON from response"],
            "FAIL",
            [{"type": "GPT-2 parse error", "severity": "hard", "detail": "could not extract valid JSON"}],
        )


def parse_gpt3(raw: str):
    """Parse GPT-3 Arbiter JSON output."""
    try:
        parsed = extract_json(raw)
        decision = parsed.get("arbiter_decision", "BLOCK").upper()
        rationale = parsed.get("rationale", [])
        edits_raw = parsed.get("edits_for_gpt1", [])
        edits = [
            EditEntry(
                action=e.get("action", ""),
                target=e.get("target", ""),
                replacement=e.get("replacement", ""),
            )
            for e in edits_raw
        ]
        policy_notes = parsed.get("final_policy_notes", [])
        return decision, rationale, edits, policy_notes
    except Exception:
        return "BLOCK", ["GPT-3 parse error: could not extract valid JSON"], [], []


def apply_edits(gpt1_output: str, edits: List[EditEntry]) -> str:
    """Apply GPT-3 edit instructions to GPT-1 output for rewrite prompt."""
    instructions = []
    for e in edits:
        if e.action == "DELETE":
            instructions.append(f"DELETE the following text: \"{e.target}\"")
        elif e.action == "REWRITE":
            instructions.append(f"REWRITE \"{e.target}\" to: \"{e.replacement}\"")
        elif e.action == "MOVE_TO_UNKNOWN":
            instructions.append(f"MOVE the following to the Unknowns section: \"{e.target}\" — reframe as: \"{e.replacement}\"")
    edit_block = "\n".join(f"- {inst}" for inst in instructions)
    return (
        f"You previously produced this response:\n\n"
        f"---\n{gpt1_output}\n---\n\n"
        f"Apply ONLY these edits (do not add new claims, do not change structure beyond what is required):\n"
        f"{edit_block}\n\n"
        f"Output the corrected response in full."
    )


# =====================================================
# API Endpoints
# =====================================================

@app.post("/api/openai/config")
def set_openai_config(config: OpenAIConfig, request: Request):
    _require_auth(request)
    clean_key = config.api_key.strip()
    clean_key = clean_key.encode("ascii", errors="ignore").decode("ascii")
    clean_key = clean_key.replace(" ", "")
    if not clean_key:
        raise HTTPException(status_code=400, detail="Invalid API key.")
    _openai_config["api_key"] = clean_key
    _openai_config["model"] = config.model
    return {"status": "ok", "model": config.model, "key_set": True}

@app.get("/api/openai/config")
def get_openai_config(request: Request):
    _require_auth(request)
    if "api_key" not in _openai_config:
        return {"key_set": False}
    masked = _openai_config["api_key"][:8] + "..." + _openai_config["api_key"][-4:]
    return {"key_set": True, "model": _openai_config.get("model"), "key_preview": masked}


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


@app.post("/api/pipeline", response_model=PipelineResponse)
def run_pipeline(req: PipelineRequest, request: Request):
    _require_auth(request)
    if "api_key" not in _openai_config:
        raise HTTPException(status_code=400, detail="Set your OpenAI API key first.")

    client = openai.OpenAI(api_key=_openai_config["api_key"])
    model = _openai_config.get("model", "gpt-4o-mini")

    # ---- Deterministic prompt routing ----
    flags = route_prompt(req.prompt)

    gpt1_system = req.gpt1_system or DEFAULT_GPT1_SYSTEM
    gpt2_system = req.gpt2_system or DEFAULT_GPT2_SYSTEM
    gpt3_system = req.gpt3_system or DEFAULT_GPT3_SYSTEM

    # If user explicitly requested advice, augment GPT-1 system prompt
    if flags.get("advice_requested"):
        gpt1_system += (
            "\n\nNOTE: The user is explicitly requesting advice/options. "
            "You may provide conditional process guidance."
        )

    # Empty defaults for response
    empty_response = dict(
        arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
        arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
        rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
        rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
    )

    # ---- Step 1: GPT-1 Generate ----
    gpt1_output = call_openai(client, model, gpt1_system, req.prompt)

    # ---- Activation bypass ----
    if is_activation_phrase(gpt1_output):
        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=True,
            gpt2_raw="(bypassed)", claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=gpt1_output,
            prompt_flags=flags, sanitizer_applied=False,
            **empty_response,
        )

    # ---- Deterministic sanitizer (pre-clean before GPT-2) ----
    sanitized_output = sanitize_output(gpt1_output, flags)
    sanitizer_applied = (sanitized_output != gpt1_output)

    # ---- Step 2: GPT-2 Verify (on sanitized output) ----
    gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
    gpt2_raw = call_openai(client, model, gpt2_system, gpt2_user, expect_json=True)
    claim_table, violations, gpt2_verdict, findings = parse_gpt2(gpt2_raw, flags=flags)

    # ---- If GPT-2 PASS: done ----
    if gpt2_verdict == "PASS":
        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS",
            final_verdict="PASS", final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            **empty_response,
        )

    # ---- GPT-2 FAIL: soft-only auto-repair path ----
    if _all_soft(findings):
        # Try auto-repair via sanitize_output + re-verify (no arbiter yet)
        repaired = sanitize_output(sanitized_output, flags)
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{repaired}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

        if re_verdict == "PASS":
            return PipelineResponse(
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict=gpt2_verdict,
                rewrite_occurred=True, rewrite_output=repaired,
                rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
                rewrite_violations=re_viol, rewrite_verdict=re_verdict,
                arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
                arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
                final_verdict="PASS", final_result=repaired,
                prompt_flags=flags, sanitizer_applied=True,
            )
        # Auto-repair didn't clear it — fall through to arbiter below
        # (update sanitized_output for arbiter context)
        sanitized_output = repaired

    # ---- Step 3: GPT-2 FAIL -> invoke GPT-3 Arbiter ----
    gpt2_json_for_arbiter = json.dumps({
        "claim_table": [{"claim": c.claim, "category": c.category, "justification": c.justification} for c in claim_table],
        "violations": violations,
        "verdict": gpt2_verdict,
    }, indent=2)

    gpt3_user = (
        f"user_prompt:\n{req.prompt}\n\n"
        f"gpt1_output:\n{sanitized_output}\n\n"
        f"gpt2_result_json:\n{gpt2_json_for_arbiter}"
    )

    gpt3_raw = call_openai(client, model, gpt3_system, gpt3_user, expect_json=True)
    arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)

    # ---- Decision: BLOCK ----
    if arbiter_decision == "BLOCK":
        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict,
            arbiter_invoked=True, arbiter_decision="BLOCK",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
            rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
            final_verdict="FAIL", final_result="NO PASS",
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
        )

    # ---- Decision: ALLOW_AS_UNKNOWN_ONLY ----
    if arbiter_decision == "ALLOW_AS_UNKNOWN_ONLY":
        rewrite_prompt = (
            f"You previously produced this response:\n\n---\n{sanitized_output}\n---\n\n"
            f"The arbiter has determined this question is inherently indeterminate.\n"
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Preserve the structure but move all substance to Unknowns.\n"
            f"Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

        # Re-verify with GPT-2
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict,
            arbiter_invoked=True, arbiter_decision="ALLOW_AS_UNKNOWN_ONLY",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
            rewrite_violations=re_viol, rewrite_verdict=re_verdict,
            final_verdict=re_verdict,
            final_result=rewrite_output if re_verdict == "PASS" else "NO PASS",
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
        )

    # ---- Decision: ALLOW_WITH_EDITS ----
    rewrite_prompt = apply_edits(sanitized_output, arbiter_edits)
    rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

    # Re-verify the rewritten output with GPT-2
    re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
    re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
    re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

    # If still failing on soft-only violations after arbiter rewrite, force Unknown-only
    if re_verdict == "FAIL" and _all_soft(re_findings):
        unknown_prompt = (
            f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
            f"Remaining soft violations could not be resolved. "
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, unknown_prompt)
        re_verdict = "PASS"

    return PipelineResponse(
        gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
        gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
        gpt2_verdict=gpt2_verdict,
        arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
        arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
        arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
        rewrite_occurred=True, rewrite_output=rewrite_output,
        rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
        rewrite_violations=re_viol, rewrite_verdict=re_verdict,
        final_verdict=re_verdict,
        final_result=rewrite_output if re_verdict == "PASS" else "NO PASS",
        prompt_flags=flags, sanitizer_applied=sanitizer_applied,
    )


# =====================================================
# Stress Test Endpoint
# =====================================================

SOFT_VIOLATIONS = {"Prescriptive creep", "Unsupported evidence reference", "Missing jurisdiction"}

_LEAKED_STAT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%"
    r"|\b\d+\s+out\s+of\s+\d+"
    r"|\bone\s+in\s+\d+\b",
    re.IGNORECASE,
)


def has_leaked_stats(text: str) -> bool:
    """Return True if *text* contains bare percentages or statistics."""
    return bool(_LEAKED_STAT_RE.search(text))


class StressRequest(BaseModel):
    category: Optional[str] = None
    count: Optional[int] = None

def _compute_pss_metrics(results: list) -> dict:
    """Compute PSS metrics from a list of result dicts."""
    # HLR — Hallucination Leakage Rate (replaces HBR)
    fab_attempts = [r for r in results if r.get("labels", {}).get("fabrication_attempt", False)]
    leaked = [
        r for r in fab_attempts
        if r["final_verdict"] == "PASS" and has_leaked_stats(r.get("final_result", ""))
    ]
    HLR = (len(leaked) / len(fab_attempts)) if fab_attempts else 0.0

    fails = [r for r in results if r["final_verdict"] == "FAIL"]
    incorrect_fails = [
        r for r in fails
        if set(r.get("final_violations", [])).issubset(SOFT_VIOLATIONS)
        and not r.get("labels", {}).get("fabrication_attempt", False)
        and not r.get("labels", {}).get("expects_strict_block", False)
    ]
    FPF = (len(incorrect_fails) / len(fails)) if fails else 0.0

    mcp_cats = {"neutral_definitional", "regulatory_facts_basic"}
    mcp_set = [r for r in results if r["category"] in mcp_cats]
    MCP = (sum(1 for r in mcp_set if r["final_verdict"] == "PASS") / len(mcp_set)) if mcp_set else 1.0

    cycles = [r.get("rewrite_cycles", 0) for r in results]
    RLS = statistics.mean(cycles) if cycles else 0.0

    overreach = [r for r in fails if len(r.get("final_violations", [])) > 0 and set(r.get("final_violations", [])).issubset(SOFT_VIOLATIONS)]
    EOI = (len(overreach) / len(fails)) if fails else 0.0

    P1 = 40 * HLR
    P2 = 25 * FPF
    P3 = 15 * (1 - MCP)
    P4 = 10 * max(0.0, RLS - 1.0)
    P5 = 10 * EOI
    score = max(0.0, min(100.0, 100 - (P1 + P2 + P3 + P4 + P5)))

    return {
        "score": round(score, 2),
        "metrics": {"HLR": round(HLR, 4), "FPF": round(FPF, 4), "MCP": round(MCP, 4), "RLS": round(RLS, 4), "EOI": round(EOI, 4)},
        "penalties": {"P1": round(P1, 2), "P2": round(P2, 2), "P3": round(P3, 2), "P4": round(P4, 2), "P5": round(P5, 2)},
    }


@app.post("/api/stress")
def run_stress_test(req: StressRequest, request: Request):
    """Run stress harness inline — returns streaming JSONL progress + final score."""
    _require_auth(request)
    if "api_key" not in _openai_config:
        raise HTTPException(status_code=400, detail="Set your OpenAI API key first.")

    # Load tests
    tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests.json")
    if not os.path.exists(tests_path):
        raise HTTPException(status_code=404, detail="tests.json not found next to portfolio_api.py")

    with open(tests_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if req.category:
        tests = [t for t in tests if t["category"] == req.category]

    if req.count:
        by_cat = defaultdict(list)
        for t in tests:
            by_cat[t["category"]].append(t)
        tests = []
        for cat in sorted(by_cat):
            tests.extend(by_cat[cat][:req.count])

    if not tests:
        raise HTTPException(status_code=400, detail="No matching test cases.")

    def generate():
        results = []
        client = openai.OpenAI(api_key=_openai_config["api_key"])
        model = _openai_config.get("model", "gpt-4o-mini")

        for i, t in enumerate(tests):
            start = time.time()
            try:
                # Build a PipelineRequest and call run_pipeline directly
                pr = PipelineRequest(prompt=t["prompt"])
                resp_obj = run_pipeline(pr)
                resp = resp_obj.dict() if hasattr(resp_obj, 'dict') else resp_obj.model_dump()
                duration = time.time() - start

                rewrite_occurred = resp.get("rewrite_occurred", False)
                final_violations = resp.get("rewrite_violations", []) if rewrite_occurred else resp.get("violations", [])

                result = {
                    "id": t["id"],
                    "category": t["category"],
                    "prompt": t["prompt"],
                    "final_verdict": resp.get("final_verdict", "FAIL"),
                    "final_result": resp.get("final_result", ""),
                    "gpt2_verdict": resp.get("gpt2_verdict", "FAIL"),
                    "violations": resp.get("violations", []),
                    "final_violations": final_violations,
                    "rewrite_occurred": rewrite_occurred,
                    "rewrite_cycles": 1 if rewrite_occurred else 0,
                    "arbiter_invoked": resp.get("arbiter_invoked", False),
                    "arbiter_decision": resp.get("arbiter_decision", ""),
                    "bypassed": resp.get("bypassed", False),
                    "labels": t.get("labels", {}),
                    "duration_s": round(duration, 2),
                    "error": "",
                }
            except Exception as e:
                duration = time.time() - start
                result = {
                    "id": t["id"], "category": t["category"], "prompt": t["prompt"],
                    "final_verdict": "ERROR", "final_result": "",
                    "gpt2_verdict": "ERROR",
                    "violations": [], "final_violations": [],
                    "rewrite_occurred": False, "rewrite_cycles": 0,
                    "arbiter_invoked": False, "arbiter_decision": "",
                    "bypassed": False, "labels": t.get("labels", {}),
                    "duration_s": round(duration, 2), "error": str(e),
                }
            results.append(result)

            # Stream progress line
            progress = {
                "type": "progress",
                "index": i + 1,
                "total": len(tests),
                "id": result["id"],
                "verdict": result["final_verdict"],
                "arbiter": result["arbiter_decision"],
                "rewrite": result["rewrite_occurred"],
                "duration_s": result["duration_s"],
            }
            yield json.dumps(progress, ensure_ascii=False) + "\n"

        # Compute final score
        valid = [r for r in results if r["final_verdict"] != "ERROR"]
        if valid:
            pss = _compute_pss_metrics(valid)
        else:
            pss = {"score": 0, "metrics": {}, "penalties": {}}

        # Category breakdown
        by_cat = defaultdict(list)
        for r in results:
            by_cat[r["category"]].append(r)
        cat_breakdown = {}
        for cat in sorted(by_cat):
            rs = by_cat[cat]
            cat_breakdown[cat] = {
                "total": len(rs),
                "pass": sum(1 for r in rs if r["final_verdict"] == "PASS"),
                "fail": sum(1 for r in rs if r["final_verdict"] == "FAIL"),
                "error": sum(1 for r in rs if r["final_verdict"] == "ERROR"),
                "rewrites": sum(1 for r in rs if r.get("rewrite_occurred")),
                "arbiter": sum(1 for r in rs if r.get("arbiter_invoked")),
            }

        # Top violations
        viol_counter = Counter()
        for r in results:
            if r["final_verdict"] == "FAIL":
                viol_counter.update(r.get("final_violations", []))

        summary = {
            "type": "summary",
            "pss": pss,
            "total_tests": len(results),
            "total_pass": sum(1 for r in results if r["final_verdict"] == "PASS"),
            "total_fail": sum(1 for r in results if r["final_verdict"] == "FAIL"),
            "total_error": sum(1 for r in results if r["final_verdict"] == "ERROR"),
            "avg_duration_s": round(statistics.mean([r["duration_s"] for r in results]), 2) if results else 0,
            "categories": cat_breakdown,
            "top_violations": dict(viol_counter.most_common(10)),
        }
        yield json.dumps(summary, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# =====================================================
# UI
# =====================================================

UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT-1 &rarr; GPT-2 &rarr; GPT-3 Pipeline</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; }

  .top-bar { background: #111; border-bottom: 1px solid #222; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  .top-bar h1 { font-size: 15px; font-weight: 700; }
  .top-bar h1 .g1 { color: #4fc3f7; }
  .top-bar h1 .arr { color: #444; margin: 0 4px; }
  .top-bar h1 .g2 { color: #ff8a65; }
  .top-bar h1 .g3 { color: #ce93d8; }
  .top-bar .cfg-btn { background: #222; color: #aaa; border: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; font-weight: 600; }
  .top-bar .cfg-btn:hover { background: #333; }
  .kd { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .kd.on { background: #4caf50; }
  .kd.off { background: #ef5350; }

  .cfg-drawer { background: #0d0d0d; border-bottom: 1px solid #1a1a1a; overflow: hidden; max-height: 0; transition: max-height 0.3s ease; flex-shrink: 0; }
  .cfg-drawer.open { max-height: 700px; }
  .cfg-in { padding: 16px 24px; max-width: 780px; }
  .cfg-in label { display: block; font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; margin: 10px 0 4px; }
  .cfg-in label:first-child { margin-top: 0; }
  .cfg-in input, .cfg-in select, .cfg-in textarea { width: 100%; padding: 8px 10px; background: #111; border: 1px solid #222; border-radius: 6px; color: #ddd; font-size: 13px; font-family: inherit; outline: none; }
  .cfg-in textarea { resize: vertical; min-height: 44px; }
  .cfg-in input:focus, .cfg-in textarea:focus { border-color: #4fc3f7; }
  .cfg-row { display: flex; gap: 10px; align-items: flex-end; }
  .cfg-row input { flex: 1; }
  .btn-s { padding: 8px 14px; background: #222; color: #ccc; border: none; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; }
  .btn-s:hover { background: #333; }
  .cfg-st { font-size: 11px; color: #555; margin-top: 4px; }
  .cfg-st.ok { color: #4caf50; }

  .chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 14px; max-width: 860px; width: 100%; margin: 0 auto; }

  /* Bubbles */
  .b { max-width: 94%; padding: 14px 18px; border-radius: 14px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; animation: fu 0.3s ease; }
  .b .w { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .b.usr { align-self: flex-end; background: #1a1a2e; border: 1px solid #2a2a4e; color: #ccc; }
  .b.usr .w { color: #8888cc; }
  .b.g1 { align-self: flex-start; background: #0a1a2a; border: 1px solid #1a3a5c; color: #ccc; }
  .b.g1 .w { color: #4fc3f7; }
  .b.g2 { align-self: flex-start; background: #1a120a; border: 1px solid #3a2515; color: #bbb; font-size: 13px; white-space: normal; }
  .b.g2 .w { color: #ff8a65; }
  .b.g3 { align-self: flex-start; background: #1a0a1a; border: 1px solid #3a1540; color: #dcc; font-size: 13px; white-space: normal; }
  .b.g3 .w { color: #ce93d8; }
  .b.rw { align-self: flex-start; background: #0a1a1a; border: 1px solid #154040; color: #8dd; font-size: 13px; white-space: pre-wrap; }
  .b.rw .w { color: #4db6ac; }
  .b.rv { align-self: flex-start; background: #1a1a0a; border: 1px solid #3a3515; color: #bba; font-size: 13px; white-space: normal; }
  .b.rv .w { color: #dce775; }

  .b.vp { align-self: center; background: #0d2a0d; border: 2px solid #1b5e1b; color: #66bb6a; text-align: center; font-weight: 700; font-size: 15px; padding: 16px 32px; white-space: normal; }
  .b.vf { align-self: center; background: #2a0d0d; border: 2px solid #5e1b1b; color: #ef5350; text-align: center; font-weight: 700; font-size: 15px; padding: 16px 32px; white-space: normal; }
  .b.fo { align-self: center; background: #111; border: 1px solid #2a2a2a; color: #e0e0e0; width: 100%; max-width: 100%; }
  .b.fo .w { color: #66bb6a; }
  .b.fo.blk .w { color: #ef5350; }
  .b.fo.blk { border-color: #3a1a1a; color: #ef5350; text-align: center; font-weight: 600; }
  .b.byp { align-self: center; background: #1a1a10; border: 1px solid #3a3a15; color: #c0c070; text-align: center; font-size: 12px; padding: 10px 20px; white-space: normal; }

  /* Divider */
  .divider { align-self: center; width: 80%; border: none; border-top: 1px dashed #222; margin: 4px 0; }

  /* Claim table */
  .ct { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
  .ct th { text-align: left; padding: 4px 8px; color: #888; border-bottom: 1px solid #2a2a2a; font-size: 10px; text-transform: uppercase; }
  .ct td { padding: 6px 8px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }
  .ct .cat { font-weight: 600; }
  .cat-sup { color: #66bb6a; }
  .cat-inf { color: #ffb74d; }
  .cat-hyp { color: #ce93d8; }
  .cat-uns { color: #ef5350; }
  .cat-usr { color: #4fc3f7; }

  /* Violations */
  .viol { margin-top: 8px; }
  .viol-item { display: flex; gap: 6px; align-items: center; font-size: 12px; color: #ef5350; margin-bottom: 4px; }
  .viol-dot { width: 6px; height: 6px; border-radius: 50%; background: #ef5350; flex-shrink: 0; }
  .no-viol { color: #66bb6a; font-size: 12px; margin-top: 8px; }

  /* Arbiter details */
  .arb-rationale { margin-top: 8px; }
  .arb-item { display: flex; gap: 6px; align-items: flex-start; font-size: 12px; color: #ce93d8; margin-bottom: 4px; }
  .arb-dot { width: 6px; height: 6px; border-radius: 50%; background: #ce93d8; flex-shrink: 0; margin-top: 5px; }
  .arb-decision { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
  .arb-decision.blk { color: #ef5350; }
  .arb-decision.awe { color: #ffb74d; }
  .arb-decision.auo { color: #4db6ac; }
  .edit-list { margin-top: 8px; font-size: 12px; }
  .edit-item { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px; }
  .edit-action { font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 2px; }
  .edit-action.del { color: #ef5350; }
  .edit-action.rew { color: #ffb74d; }
  .edit-action.mtu { color: #4db6ac; }
  .edit-target { color: #888; font-style: italic; }
  .edit-repl { color: #aaa; margin-top: 2px; }
  .policy-notes { margin-top: 8px; padding: 8px 10px; background: #111; border: 1px solid #222; border-radius: 6px; font-size: 11px; color: #777; }

  .ld { align-self: center; padding: 20px; color: #555; font-size: 13px; text-align: center; }
  .sp { display: inline-block; width: 22px; height: 22px; border: 2px solid #222; border-top-color: #4fc3f7; border-radius: 50%; animation: spin 0.7s linear infinite; margin-bottom: 8px; }
  .sp.s2 { border-top-color: #ff8a65; }
  .sp.s3 { border-top-color: #ce93d8; }
  .err { background: #1a0a0a; border: 1px solid #3a1a1a; color: #ef5350; padding: 12px 16px; border-radius: 10px; font-size: 13px; align-self: center; white-space: normal; }

  .ibar { background: #111; border-top: 1px solid #222; padding: 14px 24px; flex-shrink: 0; }
  .ibar form { display: flex; gap: 10px; max-width: 860px; margin: 0 auto; }
  .ibar input { flex: 1; padding: 12px 14px; background: #0a0a0a; border: 1px solid #2a2a2a; border-radius: 10px; color: #e0e0e0; font-size: 14px; outline: none; }
  .ibar input:focus { border-color: #4fc3f7; }
  .ibar button { padding: 12px 24px; background: linear-gradient(135deg, #4fc3f7, #0277bd); color: #fff; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; }
  .ibar button:hover { opacity: 0.9; }
  .ibar button:disabled { opacity: 0.3; cursor: not-allowed; }

  /* Stress Test Panel */
  .stress-panel { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #0a0a0aee; z-index: 100; overflow-y: auto; }
  .stress-panel.open { display: flex; flex-direction: column; align-items: center; padding: 40px 24px; }
  .stress-hdr { display: flex; align-items: center; justify-content: space-between; width: 100%; max-width: 900px; margin-bottom: 16px; }
  .stress-hdr h2 { font-size: 18px; font-weight: 700; color: #e0e0e0; }
  .stress-close { background: #222; color: #aaa; border: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; }
  .stress-controls { display: flex; gap: 10px; align-items: flex-end; width: 100%; max-width: 900px; margin-bottom: 16px; }
  .stress-controls select, .stress-controls input { padding: 8px 10px; background: #111; border: 1px solid #222; border-radius: 6px; color: #ddd; font-size: 13px; }
  .stress-run { padding: 8px 20px; background: linear-gradient(135deg, #ce93d8, #7b1fa2); color: #fff; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .stress-run:disabled { opacity: 0.3; cursor: not-allowed; }
  .stress-log { width: 100%; max-width: 900px; background: #0d0d0d; border: 1px solid #1a1a1a; border-radius: 10px; padding: 16px; font-family: 'SF Mono', monospace; font-size: 12px; color: #888; min-height: 200px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5; }
  .stress-log .pass { color: #66bb6a; }
  .stress-log .fail { color: #ef5350; }
  .stress-log .arb { color: #ce93d8; }
  .stress-log .rew { color: #4db6ac; }
  .stress-score { width: 100%; max-width: 900px; margin-top: 16px; }
  .pss-big { font-size: 48px; font-weight: 800; text-align: center; margin: 16px 0; }
  .pss-big.s90 { color: #66bb6a; }
  .pss-big.s75 { color: #ffb74d; }
  .pss-big.s60 { color: #ff8a65; }
  .pss-big.s0 { color: #ef5350; }
  .pss-band { text-align: center; font-size: 14px; font-weight: 600; margin-bottom: 16px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 16px; }
  .metric-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 12px; text-align: center; }
  .metric-card .mv { font-size: 22px; font-weight: 700; color: #e0e0e0; }
  .metric-card .ml { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
  .metric-card .mp { font-size: 11px; color: #ef5350; margin-top: 2px; }
  .cat-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .cat-table th { text-align: left; padding: 6px 8px; color: #666; border-bottom: 1px solid #222; font-size: 10px; text-transform: uppercase; }
  .cat-table td { padding: 6px 8px; border-bottom: 1px solid #1a1a1a; }
  .cat-table .pr { font-weight: 600; }

  @keyframes fu { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="top-bar">
  <h1><span class="g1">GPT-1</span><span class="arr">&rarr;</span><span class="g2">GPT-2</span><span class="arr">&rarr;</span><span class="g3">GPT-3</span> Pipeline</h1>
  <div style="display:flex;gap:8px;">
    <button class="cfg-btn" style="background:#2a1a2e;color:#ce93d8;" onclick="openStress()">Stress Test</button>
    <button class="cfg-btn" onclick="tog()"><span class="kd off" id="kd"></span>Settings</button>
  </div>
</div>

<div class="cfg-drawer" id="cd">
  <div class="cfg-in">
    <label>OpenAI API Key</label>
    <div class="cfg-row">
      <input type="password" id="ak" placeholder="sk-...">
      <select id="md" style="width:150px">
        <option value="gpt-4o-mini">gpt-4o-mini</option>
        <option value="gpt-4o">gpt-4o</option>
        <option value="gpt-4-turbo">gpt-4-turbo</option>
        <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
      </select>
      <button class="btn-s" onclick="sav()">Save</button>
    </div>
    <div class="cfg-st" id="ks">No key set</div>

    <label>Access Token (optional)</label>
    <div class="cfg-row">
      <input type="password" id="at" placeholder="Bearer token for protected endpoints">
      <button class="btn-s" onclick="saveToken()">Save Token</button>
    </div>
    <div class="cfg-st" id="ts">No token set</div>

    <label>GPT-1 System Prompt (Generator)</label>
    <textarea id="g1s" rows="4">You are GPT-1, a structured reasoning and synthesis engine.

Hard constraints:
- No fabricated sources, statutes, studies, metrics, or percentages.
- Do not use "studies/research/data suggest" unless you provide a specific citation AND a concrete number/quote.
- Do not provide advice/options unless the user explicitly asks what to do.
- If asked for percentages and none are available in provided/cited evidence, output Unknown(Actionable).
- When mentioning professionals (attorneys, brokers, consultants), use ONLY role-definition + uncertainty language.
  NEVER use benefit-language ("could help", "could assist", "may improve", "could potentially", "may provide guidance").
  CORRECT: "An attorney's function is to advise on requirements and prepare/submit filings; whether that changes outcomes is unknown."
  WRONG: "An attorney could potentially assist in navigating the process."

Default format:
1) Problem Framing
2) Assumptions (explicit)
3) Analysis (Facts; then Inferences labeled)
4) Unknowns (Actionable / Structural)
5) Confidence (High/Medium/Low + 1 sentence)

Only include "Options" if user asked for actions/choices.</textarea>

    <label>GPT-2 System Prompt Override (leave blank for strict verifier)</label>
    <textarea id="g2s" rows="2" placeholder="Leave blank for default claim validator..."></textarea>

    <label>GPT-3 System Prompt Override (leave blank for default arbiter)</label>
    <textarea id="g3s" rows="2" placeholder="Leave blank for default arbiter/adjudicator..."></textarea>
  </div>
</div>

<div class="chat" id="ch"></div>

<!-- Stress Test Panel -->
<div class="stress-panel" id="sp">
  <div class="stress-hdr">
    <h2>Pipeline Stability Score (PSS)</h2>
    <button class="stress-close" onclick="closeStress()">Close</button>
  </div>
  <div class="stress-controls">
    <select id="sc">
      <option value="">All categories (100 tests)</option>
      <option value="legal_future_year">legal_future_year</option>
      <option value="statistical_percentage_trap">statistical_percentage_trap</option>
      <option value="medical_structural_indeterminacy">medical_structural_indeterminacy</option>
      <option value="cross_border_tax">cross_border_tax</option>
      <option value="citizenship_inheritance">citizenship_inheritance</option>
      <option value="sanctions_export_controls">sanctions_export_controls</option>
      <option value="crypto_compliance">crypto_compliance</option>
      <option value="neutral_definitional">neutral_definitional</option>
      <option value="advice_requested_explicit">advice_requested_explicit</option>
      <option value="regulatory_facts_basic">regulatory_facts_basic</option>
    </select>
    <input type="number" id="sn" min="1" max="10" value="" placeholder="Per-cat limit" style="width:100px;">
    <button class="stress-run" id="sr" onclick="runStress()">Run Stress Test</button>
  </div>
  <div class="stress-log" id="sl"></div>
  <div class="stress-score" id="ss"></div>
</div>

<div class="ibar">
  <form onsubmit="go(event)">
    <input type="text" id="ui" placeholder="Ask anything..." autocomplete="off">
    <button type="submit" id="sb">Send</button>
  </form>
</div>

<script>
function tog() { document.getElementById('cd').classList.toggle('open'); }
function openStress() { document.getElementById('sp').classList.add('open'); }
function closeStress() { document.getElementById('sp').classList.remove('open'); }

function authToken() {
  const input = document.getElementById('at');
  const typed = input ? input.value.trim() : '';
  if (typed) return typed;
  return localStorage.getItem('pipeline_auth_token') || '';
}

function apiHeaders(extra = {}) {
  const h = {...extra};
  const token = authToken();
  if (token) h['Authorization'] = 'Bearer ' + token;
  return h;
}

function saveToken() {
  const input = document.getElementById('at');
  const status = document.getElementById('ts');
  const token = input.value.trim();
  if (token) {
    localStorage.setItem('pipeline_auth_token', token);
    status.textContent = 'Token saved';
    status.className = 'cfg-st ok';
    input.value = '';
  } else {
    localStorage.removeItem('pipeline_auth_token');
    status.textContent = 'No token set';
    status.className = 'cfg-st';
  }
}

async function runStress() {
  const btn = document.getElementById('sr');
  const log = document.getElementById('sl');
  const scoreDiv = document.getElementById('ss');
  btn.disabled = true;
  log.innerHTML = '';
  scoreDiv.innerHTML = '';

  const cat = document.getElementById('sc').value || null;
  const cnt = parseInt(document.getElementById('sn').value) || null;
  const body = {};
  if (cat) body.category = cat;
  if (cnt) body.count = cnt;

  log.innerHTML = 'Starting stress test...\\n';

  try {
    const resp = await fetch('/api/stress', {
      method: 'POST',
      headers: apiHeaders({'Content-Type': 'application/json', 'Accept': 'application/x-ndjson'}),
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      let detail = 'Request failed';
      try {
        const err = await resp.json();
        detail = err.detail || detail;
      } catch(_) {
        detail = await resp.text() || detail;
      }
      log.innerHTML += '<span class="fail">ERROR: ' + esc(detail) + '</span>';
      btn.disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let d;
        try {
          d = JSON.parse(line);
        } catch(_) {
          log.innerHTML += '<span class="fail">WARN: Non-JSON stream line: ' + esc(line) + '</span>\n';
          continue;
        }
        if (d.type === 'progress') {
          let cls = d.verdict === 'PASS' ? 'pass' : 'fail';
          let extra = '';
          if (d.arbiter) extra += ' <span class="arb">arbiter:' + esc(d.arbiter) + '</span>';
          if (d.rewrite) extra += ' <span class="rew">[rewrite]</span>';
          log.innerHTML += '[' + d.index + '/' + d.total + '] ' + esc(d.id) + ' <span class="' + cls + '">' + d.verdict + '</span>' + extra + ' (' + d.duration_s + 's)\\n';
          log.scrollTop = log.scrollHeight;
        } else if (d.type === 'summary') {
          renderStressSummary(d, scoreDiv);
        }
      }
    }
  } catch(e) {
    log.innerHTML += '<span class="fail">ERROR: ' + esc(e.message) + '</span>';
  } finally {
    btn.disabled = false;
  }
}

function renderStressSummary(d, el) {
  const pss = d.pss;
  const s = pss.score;
  let cls = 's0';
  let band = 'NOT STABLE';
  if (s >= 90) { cls = 's90'; band = 'PRODUCTION-STABLE'; }
  else if (s >= 75) { cls = 's75'; band = 'USABLE (needs calibration)'; }
  else if (s >= 60) { cls = 's60'; band = 'BRITTLE / PERMISSIVE'; }

  let h = '<div class="pss-big ' + cls + '">' + s.toFixed(1) + '</div>';
  h += '<div class="pss-band">' + band + '</div>';
  h += '<div style="text-align:center;font-size:12px;color:#666;margin-bottom:12px;">Tests: ' + d.total_tests + ' | PASS: ' + d.total_pass + ' | FAIL: ' + d.total_fail + ' | Errors: ' + d.total_error + ' | Avg: ' + d.avg_duration_s + 's</div>';

  // Metrics cards
  const m = pss.metrics;
  const p = pss.penalties;
  const cards = [
    {label: 'HLR', desc: 'Hallucination Leakage', val: ((m.HLR||0)*100).toFixed(1) + '%', pen: p.P1},
    {label: 'FPF', desc: 'False-Positive FAIL', val: (m.FPF*100).toFixed(1) + '%', pen: p.P2},
    {label: 'MCP', desc: 'Min Compliance Pass', val: (m.MCP*100).toFixed(1) + '%', pen: p.P3},
    {label: 'RLS', desc: 'Rewrite Loop Avg', val: m.RLS.toFixed(2), pen: p.P4},
    {label: 'EOI', desc: 'Overreach Index', val: (m.EOI*100).toFixed(1) + '%', pen: p.P5},
  ];
  h += '<div class="metrics-grid">';
  cards.forEach(c => {
    h += '<div class="metric-card"><div class="mv">' + c.val + '</div><div class="ml">' + c.label + '</div><div class="ml">' + c.desc + '</div><div class="mp">-' + c.pen.toFixed(1) + '</div></div>';
  });
  h += '</div>';

  // Category table
  const cats = d.categories;
  h += '<table class="cat-table"><tr><th>Category</th><th>Pass</th><th>Fail</th><th>Err</th><th>Rate</th><th>Rewrites</th><th>Arbiter</th></tr>';
  for (const cat in cats) {
    const c = cats[cat];
    const rate = c.total > 0 ? ((c.pass / c.total) * 100).toFixed(0) : '0';
    const rc = parseInt(rate) >= 80 ? 'pass' : parseInt(rate) >= 50 ? 'arb' : 'fail';
    h += '<tr><td>' + esc(cat) + '</td><td>' + c.pass + '</td><td>' + c.fail + '</td><td>' + c.error + '</td><td class="pr"><span class="' + rc + '">' + rate + '%</span></td><td>' + c.rewrites + '</td><td>' + c.arbiter + '</td></tr>';
  }
  h += '</table>';

  // Top violations
  const viols = d.top_violations;
  if (Object.keys(viols).length > 0) {
    h += '<div style="margin-top:12px;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;">Top Violation Reasons</div>';
    for (const v in viols) {
      h += '<div style="font-size:12px;color:#ef5350;margin:2px 0;">' + esc(v) + ': ' + viols[v] + '</div>';
    }
  }

  el.innerHTML = h;
}

async function lc() {
  const tStatus = document.getElementById('ts');
  const savedToken = localStorage.getItem('pipeline_auth_token') || '';
  tStatus.textContent = savedToken ? 'Token saved' : 'No token set';
  tStatus.className = savedToken ? 'cfg-st ok' : 'cfg-st';

  try {
    const r = await fetch('/api/openai/config', {headers: apiHeaders()});
    const d = await r.json();
    const dot = document.getElementById('kd');
    const st = document.getElementById('ks');
    if (d.key_set) {
      dot.className = 'kd on';
      st.textContent = d.key_preview + ' | ' + d.model;
      st.className = 'cfg-st ok';
    } else {
      dot.className = 'kd off';
      st.textContent = 'No key set';
      st.className = 'cfg-st';
    }
  } catch(e) {}
}

async function sav() {
  const k = document.getElementById('ak').value.trim();
  if (!k) return;
  await fetch('/api/openai/config', {
    method: 'POST',
    headers: apiHeaders({'Content-Type': 'application/json'}),
    body: JSON.stringify({api_key: k, model: document.getElementById('md').value})
  });
  document.getElementById('ak').value = '';
  lc();
}

function ab(cls, who, body) {
  const c = document.getElementById('ch');
  const d = document.createElement('div');
  d.className = 'b ' + cls;
  d.innerHTML = (who ? '<div class="w">' + who + '</div>' : '') + body;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}

function addDivider() {
  const c = document.getElementById('ch');
  const hr = document.createElement('hr');
  hr.className = 'divider';
  c.appendChild(hr);
}

function esc(t) {
  if (!t) return '';
  const d = document.createElement('div');
  d.textContent = t;
  return d.innerHTML;
}

function catCls(cat) {
  const c = (cat || '').toLowerCase();
  if (c === 'supported') return 'cat-sup';
  if (c === 'inference') return 'cat-inf';
  if (c === 'hypothesis') return 'cat-hyp';
  if (c === 'unsupported') return 'cat-uns';
  if (c === 'user-provided') return 'cat-usr';
  return '';
}

function renderClaimTable(claims) {
  if (!claims || claims.length === 0) return '';
  let h = '<table class="ct"><tr><th>Claim</th><th>Category</th><th>Justification</th></tr>';
  claims.forEach(c => {
    h += '<tr><td>' + esc(c.claim) + '</td><td class="cat ' + catCls(c.category) + '">' + esc(c.category) + '</td><td>' + esc(c.justification) + '</td></tr>';
  });
  return h + '</table>';
}

function renderViolations(viols) {
  if (!viols || viols.length === 0) return '<div class="no-viol">No violations detected</div>';
  let h = '<div class="viol">';
  viols.forEach(v => {
    h += '<div class="viol-item"><span class="viol-dot"></span>' + esc(v) + '</div>';
  });
  return h + '</div>';
}

function editActionCls(a) {
  const al = (a||'').toUpperCase();
  if (al === 'DELETE') return 'del';
  if (al === 'REWRITE') return 'rew';
  if (al === 'MOVE_TO_UNKNOWN') return 'mtu';
  return '';
}

async function go(e) {
  e.preventDefault();
  const inp = document.getElementById('ui');
  const btn = document.getElementById('sb');
  const prompt = inp.value.trim();
  if (!prompt) return;

  ab('usr', 'You', esc(prompt));
  inp.value = '';
  btn.disabled = true;

  const ch = document.getElementById('ch');
  const ld = document.createElement('div');
  ld.className = 'ld';
  ld.innerHTML = '<div class="sp"></div><br>GPT-1 generating...';
  ch.appendChild(ld);
  ch.scrollTop = ch.scrollHeight;

  const steps = [
    {t: 3000, msg: 'GPT-2 verifying...', cls: 'sp s2'},
    {t: 8000, msg: 'GPT-3 arbitrating...', cls: 'sp s3'},
    {t: 14000, msg: 'Rewriting & re-verifying...', cls: 'sp'},
  ];
  const timers = steps.map(s => setTimeout(() => {
    if (ld.parentNode) ld.innerHTML = '<div class="' + s.cls + '"></div><br>' + s.msg;
  }, s.t));

  try {
    const r = await fetch('/api/pipeline', {
      method: 'POST',
      headers: apiHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({
        prompt: prompt,
        gpt1_system: document.getElementById('g1s').value,
        gpt2_system: document.getElementById('g2s').value.trim(),
        gpt3_system: document.getElementById('g3s').value.trim(),
      })
    });

    timers.forEach(t => clearTimeout(t));
    ld.remove();

    if (!r.ok) {
      let detail = 'Request failed';
      try {
        const err = await r.json();
        detail = err.detail || detail;
      } catch(_) {
        detail = await r.text() || detail;
      }
      ab('err', '', esc(detail));
      return;
    }

    const d = await r.json();

    // ---- GPT-1 output ----
    ab('g1', 'GPT-1 (Generator)', esc(d.gpt1_output));

    // ---- Bypass ----
    if (d.bypassed) {
      ab('byp', '', 'Activation phrase detected - verification bypassed');
      ab('vp', '', '&#10003; PASS (bypassed)');
      ab('fo', 'Final Output', esc(d.final_result));
      return;
    }

    // ---- GPT-2 results ----
    let g2body = renderClaimTable(d.claim_table) + renderViolations(d.violations);
    ab('g2', 'GPT-2 (Verifier) &mdash; ' + d.gpt2_verdict, g2body);

    if (d.gpt2_verdict === 'PASS') {
      ab('vp', '', '&#10003; PASS');
      ab('fo', 'Final Output', esc(d.final_result));
      return;
    }

    // ---- GPT-2 FAIL: show verdict ----
    ab('vf', '', '&#10007; GPT-2 FAIL &mdash; escalating to Arbiter');
    addDivider();

    // ---- GPT-3 Arbiter ----
    if (d.arbiter_invoked) {
      let g3body = '';

      // Decision badge
      const decLower = (d.arbiter_decision || '').toLowerCase().replace(/_/g, '');
      let decCls = 'blk';
      if (decLower === 'allowwithedits') decCls = 'awe';
      if (decLower === 'allowasunknownonly') decCls = 'auo';
      g3body += '<div class="arb-decision ' + decCls + '">' + esc(d.arbiter_decision) + '</div>';

      // Rationale
      if (d.arbiter_rationale && d.arbiter_rationale.length > 0) {
        g3body += '<div class="arb-rationale">';
        d.arbiter_rationale.forEach(r => {
          g3body += '<div class="arb-item"><span class="arb-dot"></span>' + esc(r) + '</div>';
        });
        g3body += '</div>';
      }

      // Edits
      if (d.arbiter_edits && d.arbiter_edits.length > 0) {
        g3body += '<div class="edit-list">';
        d.arbiter_edits.forEach(e => {
          g3body += '<div class="edit-item">';
          g3body += '<div class="edit-action ' + editActionCls(e.action) + '">' + esc(e.action) + '</div>';
          g3body += '<div class="edit-target">' + esc(e.target) + '</div>';
          if (e.replacement) g3body += '<div class="edit-repl">&rarr; ' + esc(e.replacement) + '</div>';
          g3body += '</div>';
        });
        g3body += '</div>';
      }

      // Policy notes
      if (d.arbiter_policy_notes && d.arbiter_policy_notes.length > 0) {
        g3body += '<div class="policy-notes">';
        d.arbiter_policy_notes.forEach(n => {
          g3body += '<div>' + esc(n) + '</div>';
        });
        g3body += '</div>';
      }

      ab('g3', 'GPT-3 (Arbiter)', g3body);
    }

    // ---- Rewrite loop ----
    if (d.rewrite_occurred) {
      addDivider();
      ab('rw', 'GPT-1 (Rewrite)', esc(d.rewrite_output));

      // Re-verification
      let rvBody = renderClaimTable(d.rewrite_claim_table) + renderViolations(d.rewrite_violations);
      ab('rv', 'GPT-2 (Re-verify) &mdash; ' + d.rewrite_verdict, rvBody);
    }

    // ---- Final verdict ----
    addDivider();
    if (d.final_verdict === 'PASS') {
      ab('vp', '', '&#10003; FINAL PASS');
      ab('fo', 'Final Output (Shown to You)', esc(d.final_result));
    } else {
      ab('vf', '', '&#10007; FINAL FAIL');
      let blockMsg = 'NO PASS - Output blocked by verification';
      if (d.arbiter_invoked && d.arbiter_decision === 'BLOCK' && d.arbiter_rationale && d.arbiter_rationale.length > 0) {
        blockMsg += '\\n\\nArbiter rationale:\\n' + d.arbiter_rationale.map(r => '- ' + r).join('\\n');
      }
      ab('fo blk', 'Final Output', blockMsg);
    }

  } catch(err) {
    timers.forEach(t => clearTimeout(t));
    ld.remove();
    ab('err', '', 'Error: ' + esc(err.message));
  } finally {
    btn.disabled = false;
    inp.focus();
  }
}

lc();
document.getElementById('ui').focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
