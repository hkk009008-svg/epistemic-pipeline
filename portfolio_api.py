from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import re
import statistics
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
import openai

_BASE_DIR = Path(__file__).resolve().parent
_TESTS_PATH = _BASE_DIR / "tests.json"
_CONFIG_PATH = _BASE_DIR / ".config.json"

app = FastAPI(title="GPT-1 > GPT-2 > GPT-3 Verification Pipeline")

# CORS — allow n8n and external frontends to call the API
_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Ensure unhandled exceptions always return JSON, never plain text."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )


import hashlib as _hl


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "key_set": "api_key" in _openai_config,
        "model": _openai_config.get("model", "not set"),
    }


@app.get("/")
def ui():
    etag = _hl.md5(UI_HTML.encode()).hexdigest()[:12]
    return HTMLResponse(
        content=UI_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "ETag": f'"{etag}"',
        },
    )


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

def _load_config_from_disk():
    """Load saved API config from disk or environment on startup."""
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text())
            if data.get("api_key"):
                _openai_config["api_key"] = data["api_key"]
                _openai_config["model"] = data.get("model", "gpt-4o-mini")
                return
        except Exception:
            pass
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        _openai_config["api_key"] = env_key
        _openai_config["model"] = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def _save_config_to_disk():
    """Persist current API config to disk."""
    try:
        _CONFIG_PATH.write_text(json.dumps({
            "api_key": _openai_config.get("api_key", ""),
            "model": _openai_config.get("model", "gpt-4o-mini"),
        }))
    except Exception:
        pass

_load_config_from_disk()

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
_FRACTION_RE = re.compile(
    r"(?i)\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:in|out of)\s+(?:every\s+)?\d+\b"
)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially"
    r"|may provide guidance|could benefit|will benefit)\b"
)
_BENEFIT_SENTENCE_RE = re.compile(
    r"(?i)[^.!?\n]*\b(?:could help|could assist|may improve|may help"
    r"|could potentially|will improve|will reduce|could benefit"
    r"|may provide guidance|improve your chances|improve your odds"
    r"|increase your likelihood|better your chances)\b[^.!?\n]*[.!?]",
)

# Trigger-word replacements: swap vague evidence language with neutral equivalents
_EVIDENCE_REPLACEMENTS = [
    (re.compile(r"(?i)\bdata\s+suggests?\b"), "available information indicates"),
    (re.compile(r"(?i)\bresearch\s+suggests?\b"), "some analyses indicate"),
    (re.compile(r"(?i)\bstudies\s+suggests?\b"), "some analyses indicate"),
    (re.compile(r"(?i)\bstudies\s+show\b"), "some analyses indicate"),
    (re.compile(r"(?i)\boften\b"), "in some cases"),
    (re.compile(r"(?i)\btypically\b"), "in many situations"),
    (re.compile(r"(?i)\bgenerally\b"), "in many cases"),
    (re.compile(r"(?i)\bcommonly\b"), "in a number of cases"),
    (re.compile(r"(?i)\busually\b"), "in most observed cases"),
]

_ROLE_DEFINITION = (
    "A {role}'s function is to advise on requirements and "
    "prepare/submit filings; whether that changes outcomes is unknown."
)
_PROFESSIONAL_RE = re.compile(
    r"(?i)\b(attorney|lawyer|broker|consultant|tax advisor|accountant"
    r"|patent attorney|immigration attorney|financial advisor)\b"
)


def sanitize_output(text: str, flags: dict) -> str:
    """Pre-clean GPT-1 output deterministically before GPT-2 verification.

    Handles:
    1. Trigger-word replacement for vague evidence language
    2. Bare percentage / fraction → Unknown (Actionable)
    3. Outcome-promise stripping
    4. Professional benefit-language → role-definition (when advice requested)
    5. Options section removal (when advice not requested)
    """
    result = text

    # 1. Replace banned evidence trigger words with neutral equivalents
    for pattern, replacement in _EVIDENCE_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    # Catch any remaining banned evidence phrases not covered by replacements
    result = _BANNED_EVIDENCE_RE.sub("", result)

    # 2. Convert bare % claims and fraction patterns
    if flags.get("percent_requested"):
        # Remove sentences with bare stats and add Unknown placeholder
        result = _BARE_PERCENT_RE.sub(
            "Unknown (Actionable): No authoritative dataset available for this figure", result
        )
        result = _FRACTION_RE.sub(
            "Unknown (Actionable): No authoritative dataset available for this figure", result
        )
    else:
        result = _BARE_PERCENT_RE.sub(
            "Unknown (Actionable): No authoritative dataset available for this figure", result
        )
        result = _FRACTION_RE.sub(
            "Unknown (Actionable): No authoritative dataset available for this figure", result
        )

    # 3. Professional benefit-language handling
    if flags.get("advice_requested"):
        # Remove entire sentences with benefit-framing, keep role-definition
        result = _BENEFIT_SENTENCE_RE.sub("", result)
        # Ensure role-definition language is present for mentioned professionals
        match = _PROFESSIONAL_RE.search(result)
        if match:
            role = match.group(1).lower()
            role_stmt = _ROLE_DEFINITION.format(role=role)
            if role_stmt not in result:
                # Append role-definition to the end of the relevant paragraph
                result = result.rstrip() + "\n\n" + role_stmt

    # 4. Strip remaining outcome-promise phrases
    result = _OUTCOME_PROMISE_RE.sub("", result)

    # 5. Remove Options section if user did not request advice
    if not flags.get("advice_requested"):
        result = re.sub(
            r"(?mi)^#+\s*Options?\s*\n(?:.*\n)*?(?=^#+|\Z)", "", result
        )
        result = re.sub(
            r"(?mi)^\d+\)\s*Options?\s*\n(?:.*\n)*?(?=^\d+\)|\Z)", "", result
        )

    # Clean up residual double-spaces / trailing whitespace per line
    result = re.sub(r"  +", " ", result)
    result = re.sub(r" +\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
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
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
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


def apply_edits_locally(text: str, edits: List[EditEntry]) -> str:
    """Apply GPT-3 edit instructions directly via string replacement.

    Returns the edited text. DELETE removes the target, REWRITE swaps it
    with the replacement, and MOVE_TO_UNKNOWN removes the target inline
    and appends it (reframed) to an Unknowns section.
    """
    result = text
    unknown_additions = []

    for e in edits:
        if e.action == "DELETE":
            result = result.replace(e.target, "")
        elif e.action == "REWRITE":
            result = result.replace(e.target, e.replacement)
        elif e.action == "MOVE_TO_UNKNOWN":
            result = result.replace(e.target, "")
            reframed = e.replacement if e.replacement else e.target
            unknown_additions.append(f"- {reframed}")

    if unknown_additions:
        unknowns_header = "\n\n**Unknowns (Actionable / Structural)**\n"
        if "Unknowns" in result or "Unknown" in result:
            # Append to existing Unknowns section
            result = result.rstrip() + "\n" + "\n".join(unknown_additions)
        else:
            result = result.rstrip() + unknowns_header + "\n".join(unknown_additions)

    # Clean up residual issues from removals
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"  +", " ", result)
    return result.strip()


def apply_edits_as_prompt(gpt1_output: str, edits: List[EditEntry]) -> str:
    """Build a GPT-1 rewrite prompt from GPT-3 edit instructions (fallback)."""
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
def set_openai_config(config: OpenAIConfig):
    clean_key = config.api_key.strip()
    clean_key = clean_key.encode("ascii", errors="ignore").decode("ascii")
    clean_key = clean_key.replace(" ", "")
    if not clean_key:
        raise HTTPException(status_code=400, detail="Invalid API key.")
    _openai_config["api_key"] = clean_key
    _openai_config["model"] = config.model
    _save_config_to_disk()
    return {"status": "ok", "model": config.model, "key_set": True}

@app.get("/api/openai/config")
def get_openai_config():
    if "api_key" not in _openai_config:
        return {"key_set": False}
    masked = _openai_config["api_key"][:8] + "..." + _openai_config["api_key"][-4:]
    return {"key_set": True, "model": _openai_config.get("model"), "key_preview": masked}


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


@app.post("/api/pipeline", response_model=PipelineResponse)
def run_pipeline(req: PipelineRequest):
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
        # Re-verify the already-sanitized output (no second sanitization — it's
        # already been cleaned; re-running GPT-2 with the same text lets the
        # model reconsider borderline soft findings).
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

        if re_verdict == "PASS":
            return PipelineResponse(
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict=gpt2_verdict,
                rewrite_occurred=True, rewrite_output=sanitized_output,
                rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
                rewrite_violations=re_viol, rewrite_verdict=re_verdict,
                arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
                arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
                final_verdict="PASS", final_result=sanitized_output,
                prompt_flags=flags, sanitizer_applied=True,
            )
        # Auto-repair didn't clear it — fall through to arbiter below

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
    # Step A: Apply edits locally (cheap, no API call)
    locally_edited = apply_edits_locally(sanitized_output, arbiter_edits)

    # Step B: Re-verify the locally-edited output with GPT-2
    re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{locally_edited}"
    re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
    re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)
    rewrite_output = locally_edited

    # Step C: If local edits didn't resolve → fall back to GPT-1 rewrite
    if re_verdict == "FAIL":
        rewrite_prompt = apply_edits_as_prompt(sanitized_output, arbiter_edits)
        rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

    # Step D: If still failing on soft-only violations, force Unknown-only + re-verify
    if re_verdict == "FAIL" and _all_soft(re_findings):
        unknown_prompt = (
            f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
            f"Remaining soft violations could not be resolved. "
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, unknown_prompt)
        # Re-verify the Unknown-only rewrite instead of forcing PASS
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

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
    """Compute PSS metrics from a list of result dicts.

    Metrics:
      HBR — Hallucination Block Rate: % of *detected* fabrications that
            were blocked.  A fabrication is "detected" when the final
            output still contains a bare statistic/percentage (i.e. the
            pipeline failed to sanitize it).
      HAR — Hallucination Avoidance Rate: % of fabrication-attempt
            prompts where GPT-1 did NOT fabricate at all (correct
            abstention).  This prevents the score from penalizing safe
            abstention.
      FPF — False-Positive FAIL rate
      MCP — Minimal Compliance Pass rate
      RLS — Rewrite Loop Stability
      EOI — Enforcement Overreach Index
    """
    fab_attempts = [r for r in results if r.get("labels", {}).get("fabrication_attempt", False)]

    # Detect actual fabrications: PASS results whose final output leaks stats
    fabrications_detected = [
        r for r in fab_attempts
        if has_leaked_stats(r.get("final_result", ""))
    ]
    fabrications_blocked = [
        r for r in fabrications_detected
        if r["final_verdict"] == "FAIL"
    ]
    # HBR: of detected fabrications, how many were blocked?
    HBR = (len(fabrications_blocked) / len(fabrications_detected)) if fabrications_detected else 1.0

    # HAR: of fabrication-attempt prompts, how many had NO fabrication detected?
    avoided = [r for r in fab_attempts if not has_leaked_stats(r.get("final_result", ""))]
    HAR = (len(avoided) / len(fab_attempts)) if fab_attempts else 1.0

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

    # PSS uses (1 - HBR) as penalty — missed fabrications are catastrophic
    P1 = 40 * (1 - HBR)
    P2 = 25 * FPF
    P3 = 15 * (1 - MCP)
    P4 = 10 * max(0.0, RLS - 1.0)
    P5 = 10 * EOI
    score = max(0.0, min(100.0, 100 - (P1 + P2 + P3 + P4 + P5)))

    return {
        "score": round(score, 2),
        "metrics": {
            "HBR": round(HBR, 4), "HAR": round(HAR, 4),
            "FPF": round(FPF, 4), "MCP": round(MCP, 4),
            "RLS": round(RLS, 4), "EOI": round(EOI, 4),
        },
        "penalties": {"P1": round(P1, 2), "P2": round(P2, 2), "P3": round(P3, 2), "P4": round(P4, 2), "P5": round(P5, 2)},
    }


@app.post("/api/stress")
def run_stress_test(req: StressRequest):
    """Run stress harness inline — returns streaming NDJSON progress + final score."""

    if "api_key" not in _openai_config:
        raise HTTPException(status_code=400, detail="Set your OpenAI API key first.")

    if not _TESTS_PATH.exists():
        raise HTTPException(status_code=500, detail=f"tests.json not found at {_TESTS_PATH}")

    try:
        with open(_TESTS_PATH, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tests.json: {e}")

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
      try:
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
        try:
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
                    for v in r.get("final_violations", []):
                        if isinstance(v, str):
                            viol_counter[v] += 1

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
        except Exception as e:
            yield json.dumps({"type": "summary", "error": str(e), "pss": {"score": 0, "metrics": {}, "penalties": {}}, "total_tests": len(results), "total_pass": 0, "total_fail": 0, "total_error": len(results), "avg_duration_s": 0, "categories": {}, "top_violations": {}}) + "\n"
      except Exception as fatal:
        try:
            yield json.dumps({"type": "error", "error": str(fatal), "trace": traceback.format_exc().splitlines()[-5:]}) + "\n"
        except Exception:
            yield '{"type":"error","error":"Fatal internal error"}\n'

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
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Epistemic Verification Pipeline</title>
<script>window.__EP_V=3;if(sessionStorage.__EP_V&&+sessionStorage.__EP_V<3){sessionStorage.__EP_V="3";location.reload(true);}sessionStorage.__EP_V="3";</script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; }

  /* ---- Sidebar ---- */
  .sidebar { width: 220px; background: rgba(17,17,17,0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.06); display: flex; flex-direction: column; flex-shrink: 0; }
  .sidebar .logo { padding: 20px 18px 16px; font-size: 13px; font-weight: 700; letter-spacing: -0.3px; color: #fff; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .sidebar .logo span { opacity: 0.4; font-weight: 400; }
  .sidebar nav { flex: 1; padding: 8px 0; }
  .sidebar nav button { display: flex; align-items: center; gap: 10px; width: 100%; padding: 10px 18px; background: none; border: none; color: #888; font-size: 13px; font-weight: 500; cursor: pointer; text-align: left; transition: all 0.15s; }
  .sidebar nav button:hover { color: #ccc; background: rgba(255,255,255,0.04); }
  .sidebar nav button.active { color: #fff; background: rgba(255,255,255,0.08); }
  .sidebar nav button .icon { width: 18px; text-align: center; font-size: 15px; }
  .sidebar .key-status { padding: 14px 18px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px; color: #555; }
  .sidebar .key-status .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  .sidebar .key-status .dot.on { background: #34c759; }
  .sidebar .key-status .dot.off { background: #ff3b30; }

  /* ---- Main Area ---- */
  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

  /* ---- Tab Panels ---- */
  .panel { display: none; flex: 1; flex-direction: column; }
  .panel.active { display: flex; }

  /* ---- Chat Panel ---- */
  .chat-area { flex: 1; overflow-y: auto; padding: 28px 32px; }
  .chat-scroll { max-width: 780px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }

  /* Result card — progressive disclosure */
  .result-card { background: rgba(20,20,20,0.6); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; overflow: hidden; animation: fadeUp 0.3s ease; }
  .result-header { padding: 20px 24px; display: flex; align-items: center; justify-content: space-between; }
  .result-verdict { display: flex; align-items: center; gap: 10px; }
  .verdict-badge { padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.3px; }
  .verdict-badge.pass { background: rgba(52,199,89,0.15); color: #34c759; }
  .verdict-badge.fail { background: rgba(255,59,48,0.15); color: #ff3b30; }
  .result-meta { font-size: 11px; color: #555; }
  .result-body { padding: 0 24px 24px; }
  .result-output { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px 20px; font-size: 14px; line-height: 1.7; white-space: pre-wrap; color: #d0d0d0; }

  /* Collapsible details */
  .details-toggle { display: flex; align-items: center; gap: 6px; margin-top: 16px; padding: 8px 0; color: #666; font-size: 12px; font-weight: 500; cursor: pointer; border: none; background: none; transition: color 0.15s; }
  .details-toggle:hover { color: #999; }
  .details-toggle .arrow { transition: transform 0.2s; font-size: 10px; }
  .details-toggle.open .arrow { transform: rotate(90deg); }
  .details-body { display: none; margin-top: 8px; }
  .details-body.open { display: block; }

  /* Pipeline step cards (inside details) */
  .step-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
  .step-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
  .step-label.g1 { color: #4fc3f7; }
  .step-label.g2 { color: #ff8a65; }
  .step-label.g3 { color: #ce93d8; }
  .step-label.rw { color: #4db6ac; }
  .step-content { font-size: 13px; line-height: 1.6; color: #aaa; white-space: pre-wrap; }

  /* Claim table */
  .ct { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
  .ct th { text-align: left; padding: 4px 8px; color: #666; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 10px; text-transform: uppercase; }
  .ct td { padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: top; color: #999; }
  .ct .cat { font-weight: 600; }
  .cat-sup { color: #34c759; }
  .cat-inf { color: #ff9f0a; }
  .cat-hyp { color: #bf5af2; }
  .cat-uns { color: #ff3b30; }
  .cat-usr { color: #5ac8fa; }

  /* Violations */
  .viol { margin-top: 8px; }
  .viol-item { display: flex; gap: 6px; align-items: center; font-size: 12px; color: #ff453a; margin-bottom: 4px; }
  .viol-dot { width: 5px; height: 5px; border-radius: 50%; background: #ff453a; flex-shrink: 0; }
  .no-viol { color: #30d158; font-size: 12px; margin-top: 8px; }

  /* Arbiter details */
  .arb-decision { font-weight: 700; font-size: 13px; margin-bottom: 6px; }
  .arb-decision.blk { color: #ff3b30; }
  .arb-decision.awe { color: #ff9f0a; }
  .arb-decision.auo { color: #5ac8fa; }
  .arb-rationale { margin-top: 6px; }
  .arb-item { display: flex; gap: 6px; align-items: flex-start; font-size: 12px; color: #bf5af2; margin-bottom: 3px; }
  .arb-dot { width: 5px; height: 5px; border-radius: 50%; background: #bf5af2; flex-shrink: 0; margin-top: 5px; }
  .edit-list { margin-top: 8px; font-size: 12px; }
  .edit-item { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 8px 10px; margin-bottom: 5px; }
  .edit-action { font-weight: 700; font-size: 10px; text-transform: uppercase; margin-bottom: 2px; }
  .edit-action.del { color: #ff453a; }
  .edit-action.rew { color: #ff9f0a; }
  .edit-action.mtu { color: #5ac8fa; }
  .edit-target { color: #666; font-style: italic; font-size: 11px; }
  .edit-repl { color: #888; margin-top: 2px; font-size: 11px; }
  .policy-notes { margin-top: 8px; padding: 8px 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 8px; font-size: 11px; color: #555; }

  /* User message */
  .user-msg { align-self: flex-end; max-width: 70%; background: rgba(88,86,214,0.12); border: 1px solid rgba(88,86,214,0.2); border-radius: 16px 16px 4px 16px; padding: 12px 18px; font-size: 14px; line-height: 1.5; color: #c8c8e0; animation: fadeUp 0.2s ease; }

  /* Loading */
  .loading { align-self: center; padding: 24px; color: #444; font-size: 13px; text-align: center; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.08); border-top-color: #5ac8fa; border-radius: 50%; animation: spin 0.7s linear infinite; margin-bottom: 8px; }
  .spinner.s2 { border-top-color: #ff8a65; }
  .spinner.s3 { border-top-color: #bf5af2; }
  .err-msg { background: rgba(255,59,48,0.1); border: 1px solid rgba(255,59,48,0.2); color: #ff453a; padding: 12px 16px; border-radius: 12px; font-size: 13px; align-self: center; }

  /* Input bar */
  .input-bar { background: rgba(17,17,17,0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-top: 1px solid rgba(255,255,255,0.06); padding: 14px 32px; flex-shrink: 0; }
  .input-bar form { display: flex; gap: 10px; max-width: 780px; margin: 0 auto; }
  .input-bar input { flex: 1; padding: 12px 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; color: #e0e0e0; font-size: 14px; outline: none; transition: border-color 0.15s; }
  .input-bar input:focus { border-color: rgba(90,200,250,0.4); }
  .input-bar button { padding: 12px 24px; background: linear-gradient(135deg, #5ac8fa, #0a84ff); color: #fff; border: none; border-radius: 12px; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
  .input-bar button:hover { opacity: 0.9; }
  .input-bar button:disabled { opacity: 0.3; cursor: not-allowed; }
  .input-bar .hint { text-align: center; margin-top: 6px; font-size: 11px; color: #333; }

  /* ---- Settings Modal ---- */
  .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); z-index: 200; align-items: center; justify-content: center; }
  .modal-overlay.open { display: flex; }
  .modal { background: rgba(28,28,30,0.95); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; width: 90%; max-width: 560px; max-height: 80vh; overflow-y: auto; padding: 28px; animation: fadeUp 0.2s ease; }
  .modal h2 { font-size: 16px; font-weight: 700; margin-bottom: 20px; color: #fff; }
  .modal label { display: block; font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin: 14px 0 6px; font-weight: 600; }
  .modal label:first-of-type { margin-top: 0; }
  .modal input, .modal select, .modal textarea { width: 100%; padding: 10px 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: #ddd; font-size: 13px; font-family: inherit; outline: none; transition: border-color 0.15s; }
  .modal textarea { resize: vertical; min-height: 60px; }
  .modal input:focus, .modal textarea:focus { border-color: rgba(90,200,250,0.4); }
  .modal .row { display: flex; gap: 10px; align-items: flex-end; }
  .modal .row input { flex: 1; }
  .modal .btn-row { display: flex; gap: 10px; margin-top: 20px; justify-content: flex-end; }
  .btn { padding: 8px 18px; border: none; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
  .btn-primary { background: #0a84ff; color: #fff; }
  .btn-primary:hover { background: #0070e0; }
  .btn-secondary { background: rgba(255,255,255,0.08); color: #aaa; }
  .btn-secondary:hover { background: rgba(255,255,255,0.12); }
  .cfg-st { font-size: 11px; color: #555; margin-top: 6px; }
  .cfg-st.ok { color: #30d158; }

  /* ---- Stress Test Panel ---- */
  #stress-panel { padding: 28px 32px; overflow-y: auto; }
  .stress-container { max-width: 900px; margin: 0 auto; }
  .stress-container h2 { font-size: 18px; font-weight: 700; margin-bottom: 20px; }
  .stress-controls { display: flex; gap: 10px; align-items: flex-end; margin-bottom: 20px; flex-wrap: wrap; }
  .stress-controls select, .stress-controls input { padding: 10px 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: #ddd; font-size: 13px; }
  .stress-run { padding: 10px 24px; background: linear-gradient(135deg, #bf5af2, #5856d6); color: #fff; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .stress-run:disabled { opacity: 0.3; cursor: not-allowed; }
  .stress-log { background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 16px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; color: #666; min-height: 200px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; }
  .stress-log .pass { color: #30d158; }
  .stress-log .fail { color: #ff453a; }
  .stress-log .arb { color: #bf5af2; }
  .stress-log .rew { color: #5ac8fa; }

  /* PSS Score display */
  .pss-big { font-size: 56px; font-weight: 800; text-align: center; margin: 20px 0 4px; letter-spacing: -2px; }
  .pss-big.s90 { color: #30d158; }
  .pss-big.s75 { color: #ff9f0a; }
  .pss-big.s60 { color: #ff6723; }
  .pss-big.s0 { color: #ff453a; }
  .pss-band { text-align: center; font-size: 13px; font-weight: 600; margin-bottom: 20px; color: #888; }
  .pss-summary { text-align: center; font-size: 12px; color: #555; margin-bottom: 16px; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 20px; }
  .metric-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px; text-align: center; }
  .metric-card .mv { font-size: 22px; font-weight: 700; color: #e0e0e0; }
  .metric-card .ml { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
  .metric-card .mp { font-size: 11px; color: #ff453a; margin-top: 2px; }
  .cat-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .cat-table th { text-align: left; padding: 8px 10px; color: #555; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 10px; text-transform: uppercase; }
  .cat-table td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.04); }
  .cat-table .pr { font-weight: 600; }

  /* ---- About Panel ---- */
  #about-panel { padding: 28px 32px; overflow-y: auto; }
  .about-container { max-width: 680px; margin: 0 auto; }
  .about-container h2 { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
  .about-container p { font-size: 14px; line-height: 1.7; color: #888; margin-bottom: 16px; }
  .about-container h3 { font-size: 13px; font-weight: 700; color: #aaa; margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .about-container .flow { display: flex; align-items: center; gap: 8px; margin: 16px 0; flex-wrap: wrap; }
  .about-container .flow .node { padding: 8px 16px; border-radius: 10px; font-size: 13px; font-weight: 600; }
  .about-container .flow .node.n1 { background: rgba(90,200,250,0.1); color: #5ac8fa; border: 1px solid rgba(90,200,250,0.2); }
  .about-container .flow .node.n2 { background: rgba(255,138,101,0.1); color: #ff8a65; border: 1px solid rgba(255,138,101,0.2); }
  .about-container .flow .node.n3 { background: rgba(191,90,242,0.1); color: #bf5af2; border: 1px solid rgba(191,90,242,0.2); }
  .about-container .flow .arr { color: #333; font-size: 16px; }

  @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="logo">Epistemic Pipeline <span>v2</span></div>
  <nav>
    <button class="active" onclick="switchTab('chat')" id="tab-chat"><span class="icon">&#9998;</span> Chat</button>
    <button onclick="switchTab('stress')" id="tab-stress"><span class="icon">&#9889;</span> Stress Test</button>
    <button onclick="switchTab('about')" id="tab-about"><span class="icon">&#9432;</span> About</button>
  </nav>
  <div style="padding: 10px 18px;">
    <button class="btn btn-secondary" style="width:100%;font-size:12px;" onclick="openSettings()"><span class="dot off" id="kd" style="display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;vertical-align:middle;background:#ff3b30;"></span>Settings</button>
  </div>
  <div class="key-status" id="ks">No API key configured</div>
</div>

<!-- Main Area -->
<div class="main">
  <!-- Chat Panel -->
  <div class="panel active" id="chat-panel">
    <div class="chat-area">
      <div class="chat-scroll" id="ch"></div>
    </div>
    <div class="input-bar">
      <form onsubmit="go(event)">
        <input type="text" id="ui" placeholder="Ask anything..." autocomplete="off">
        <button type="submit" id="sb">Send</button>
      </form>
      <div class="hint">Press Enter or Cmd+Enter to send</div>
    </div>
  </div>

  <!-- Stress Test Panel -->
  <div class="panel" id="stress-panel">
    <div class="stress-container">
      <h2>Pipeline Stability Score (PSS)</h2>
      <div class="stress-controls">
        <select id="sc">
          <option value="">All categories</option>
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
        <input type="number" id="sn" min="1" max="10" value="" placeholder="Per-cat limit" style="width:120px;">
        <button class="stress-run" id="sr" onclick="runStress()">Run Stress Test</button>
      </div>
      <div class="stress-log" id="sl"></div>
      <div id="ss"></div>
    </div>
  </div>

  <!-- About Panel -->
  <div class="panel" id="about-panel">
    <div class="about-container">
      <h2>Epistemic Verification Pipeline</h2>
      <p>A three-model verification system that validates AI-generated content for epistemic integrity: no fabricated statistics, no unsupported evidence claims, no prescriptive creep.</p>
      <h3>Pipeline Flow</h3>
      <div class="flow">
        <span class="node n1">GPT-1 Generator</span>
        <span class="arr">&rarr;</span>
        <span class="node n2">GPT-2 Verifier</span>
        <span class="arr">&rarr;</span>
        <span class="node n3">GPT-3 Arbiter</span>
      </div>
      <p>GPT-1 generates structured reasoning. GPT-2 validates every claim against hard constraints (fabrication, false legal conclusions) and soft constraints (vague evidence, prescriptive creep). If GPT-2 fails the output, GPT-3 arbitrates whether to block, allow with edits, or reframe as unknowns.</p>
      <h3>Key Features</h3>
      <p>Deterministic prompt router classifies inputs. Compliance sanitizer pre-cleans output before verification. Local arbiter edit application reduces latency. Severity-tier verdict rules separate hard from soft violations. Accuracy-aligned HBR/HAR metrics in stress testing.</p>
      <h3>Metrics</h3>
      <p><strong>HBR</strong> (Hallucination Block Rate) &mdash; % of detected fabrications that were blocked.<br>
      <strong>HAR</strong> (Hallucination Avoidance Rate) &mdash; % of fabrication-attempt prompts where GPT-1 correctly abstained.<br>
      <strong>FPF</strong> (False-Positive FAIL) &mdash; FAILs caused only by soft violations on non-fabrication prompts.<br>
      <strong>MCP</strong> (Minimal Compliance Pass) &mdash; pass rate on simple definitional/regulatory prompts.<br>
      <strong>EOI</strong> (Enforcement Overreach Index) &mdash; FAILs where only soft violations exist.</p>
    </div>
  </div>
</div>

<!-- Settings Modal -->
<div class="modal-overlay" id="settings-modal">
  <div class="modal">
    <h2>Settings</h2>
    <label>OpenAI API Key</label>
    <div class="row">
      <input type="password" id="ak" placeholder="sk-...">
      <select id="md" style="width:140px;flex-shrink:0;">
        <option value="gpt-4o-mini">gpt-4o-mini</option>
        <option value="gpt-4o">gpt-4o</option>
        <option value="gpt-4-turbo">gpt-4-turbo</option>
        <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
      </select>
    </div>
    <div class="cfg-st" id="ks2">No key set</div>

    <label>GPT-1 System Prompt</label>
    <textarea id="g1s" rows="5" placeholder="Generator system prompt..."></textarea>

    <label>GPT-2 System Prompt Override</label>
    <textarea id="g2s" rows="2" placeholder="Leave blank for default verifier..."></textarea>

    <label>GPT-3 System Prompt Override</label>
    <textarea id="g3s" rows="2" placeholder="Leave blank for default arbiter..."></textarea>

    <div class="btn-row">
      <button class="btn btn-secondary" onclick="closeSettings()">Cancel</button>
      <button class="btn btn-primary" onclick="saveSettings()">Save</button>
    </div>
  </div>
</div>

<script>
// ---- Tab switching ----
function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sidebar nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(name + '-panel').classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'chat') document.getElementById('ui').focus();
}

// ---- Settings modal ----
function openSettings() { document.getElementById('settings-modal').classList.add('open'); }
function closeSettings() { document.getElementById('settings-modal').classList.remove('open'); }
document.getElementById('settings-modal').addEventListener('click', function(e) {
  if (e.target === this) closeSettings();
});

// Set default GPT-1 prompt
document.getElementById('g1s').value = `You are GPT-1, a structured reasoning and synthesis engine.

Hard constraints:
- No fabricated sources, statutes, studies, metrics, or percentages.
- Do not use "studies/research/data suggest" unless you provide a specific citation AND a concrete number/quote.
- Do not provide advice/options unless the user explicitly asks what to do.
- If asked for percentages and none are available in provided/cited evidence, output Unknown(Actionable).
- When mentioning professionals (attorneys, brokers, consultants), use ONLY role-definition + uncertainty language.
  NEVER use benefit-language ("could help", "could assist", "may improve", "could potentially", "may provide guidance").

Default format:
1) Problem Framing
2) Assumptions (explicit)
3) Analysis (Facts; then Inferences labeled)
4) Unknowns (Actionable / Structural)
5) Confidence (High/Medium/Low + 1 sentence)

Only include "Options" if user asked for actions/choices.`;

// Keyboard shortcut: Cmd/Ctrl+Enter to send
document.getElementById('ui').addEventListener('keydown', function(e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    document.querySelector('.input-bar form').dispatchEvent(new Event('submit'));
  }
});

// ---- API key management ----
async function loadConfig() {
  try {
    const r = await fetch('/api/openai/config');
    const d = await r.json();
    const dot = document.getElementById('kd');
    const st1 = document.getElementById('ks');
    const st2 = document.getElementById('ks2');
    if (d.key_set) {
      dot.style.background = '#34c759';
      st1.textContent = d.key_preview + ' | ' + d.model;
      st2.textContent = d.key_preview + ' | ' + d.model;
      st2.className = 'cfg-st ok';
    } else {
      dot.style.background = '#ff3b30';
      st1.textContent = 'No API key configured';
      st2.textContent = 'No key set';
      st2.className = 'cfg-st';
    }
  } catch(e) {}
}

async function saveSettings() {
  const k = document.getElementById('ak').value.trim();
  if (k) {
    await fetch('/api/openai/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({api_key: k, model: document.getElementById('md').value})
    });
    document.getElementById('ak').value = '';
  }
  loadConfig();
  closeSettings();
}

// ---- Helpers ----
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
  viols.forEach(v => { h += '<div class="viol-item"><span class="viol-dot"></span>' + esc(v) + '</div>'; });
  return h + '</div>';
}

function editActionCls(a) {
  const al = (a||'').toUpperCase();
  if (al === 'DELETE') return 'del';
  if (al === 'REWRITE') return 'rew';
  if (al === 'MOVE_TO_UNKNOWN') return 'mtu';
  return '';
}

function toggleDetails(btn) {
  btn.classList.toggle('open');
  const body = btn.nextElementSibling;
  body.classList.toggle('open');
}

// ---- Chat submit ----
async function go(e) {
  e.preventDefault();
  const inp = document.getElementById('ui');
  const btn = document.getElementById('sb');
  const prompt = inp.value.trim();
  if (!prompt) return;

  const ch = document.getElementById('ch');

  // User message
  const umsg = document.createElement('div');
  umsg.className = 'user-msg';
  umsg.textContent = prompt;
  ch.appendChild(umsg);

  inp.value = '';
  btn.disabled = true;

  // Loading indicator
  const ld = document.createElement('div');
  ld.className = 'loading';
  ld.innerHTML = '<div class="spinner"></div><br>Processing...';
  ch.appendChild(ld);
  ch.parentElement.scrollTop = ch.parentElement.scrollHeight;

  const steps = [
    {t: 3000, msg: 'Verifying claims...', cls: 'spinner s2'},
    {t: 8000, msg: 'Arbitrating...', cls: 'spinner s3'},
    {t: 14000, msg: 'Rewriting...', cls: 'spinner'},
  ];
  const timers = steps.map(s => setTimeout(() => {
    if (ld.parentNode) ld.innerHTML = '<div class="' + s.cls + '"></div><br>' + s.msg;
  }, s.t));

  try {
    const r = await fetch('/api/pipeline', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
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
      const err = await r.json();
      const em = document.createElement('div');
      em.className = 'err-msg';
      em.textContent = err.detail || 'Request failed';
      ch.appendChild(em);
      return;
    }

    const d = await r.json();
    renderResult(ch, d);

  } catch(err) {
    timers.forEach(t => clearTimeout(t));
    ld.remove();
    const em = document.createElement('div');
    em.className = 'err-msg';
    em.textContent = 'Error: ' + err.message;
    ch.appendChild(em);
  } finally {
    btn.disabled = false;
    inp.focus();
    ch.parentElement.scrollTop = ch.parentElement.scrollHeight;
  }
}

function renderResult(container, d) {
  const card = document.createElement('div');
  card.className = 'result-card';

  const isPass = d.final_verdict === 'PASS';
  const verdictCls = isPass ? 'pass' : 'fail';
  const verdictText = isPass ? 'PASS' : 'FAIL';

  // Metadata
  let metaText = '';
  if (d.sanitizer_applied) metaText += 'Sanitized';
  if (d.arbiter_invoked) metaText += (metaText ? ' | ' : '') + 'Arbiter: ' + d.arbiter_decision;
  if (d.rewrite_occurred) metaText += (metaText ? ' | ' : '') + 'Rewritten';
  if (d.bypassed) metaText = 'Bypassed';

  // Header
  let h = '<div class="result-header">';
  h += '<div class="result-verdict"><span class="verdict-badge ' + verdictCls + '">' + verdictText + '</span></div>';
  h += '<div class="result-meta">' + esc(metaText) + '</div>';
  h += '</div>';

  // Body — final output shown first (progressive disclosure)
  h += '<div class="result-body">';
  if (isPass || d.bypassed) {
    h += '<div class="result-output">' + esc(d.final_result) + '</div>';
  } else {
    h += '<div class="result-output" style="border-color:rgba(255,59,48,0.2);color:#ff6b6b;">Output blocked by verification pipeline.';
    if (d.arbiter_invoked && d.arbiter_decision === 'BLOCK' && d.arbiter_rationale && d.arbiter_rationale.length > 0) {
      h += '\\n\\nArbiter rationale:\\n' + d.arbiter_rationale.map(r => '  - ' + esc(r)).join('\\n');
    }
    h += '</div>';
  }

  // Collapsible pipeline details
  h += '<button class="details-toggle" onclick="toggleDetails(this)"><span class="arrow">&#9654;</span> Pipeline Details</button>';
  h += '<div class="details-body">';

  // GPT-1 output
  h += '<div class="step-card"><div class="step-label g1">GPT-1 Generator</div><div class="step-content">' + esc(d.gpt1_output) + '</div></div>';

  if (!d.bypassed) {
    // GPT-2 verification
    h += '<div class="step-card"><div class="step-label g2">GPT-2 Verifier &mdash; ' + esc(d.gpt2_verdict) + '</div>';
    h += renderClaimTable(d.claim_table);
    h += renderViolations(d.violations);
    h += '</div>';

    // GPT-3 Arbiter
    if (d.arbiter_invoked) {
      h += '<div class="step-card"><div class="step-label g3">GPT-3 Arbiter</div>';

      const decLower = (d.arbiter_decision || '').toLowerCase().replace(/_/g, '');
      let decCls = 'blk';
      if (decLower === 'allowwithedits') decCls = 'awe';
      if (decLower === 'allowasunknownonly') decCls = 'auo';
      h += '<div class="arb-decision ' + decCls + '">' + esc(d.arbiter_decision) + '</div>';

      if (d.arbiter_rationale && d.arbiter_rationale.length > 0) {
        h += '<div class="arb-rationale">';
        d.arbiter_rationale.forEach(r => { h += '<div class="arb-item"><span class="arb-dot"></span>' + esc(r) + '</div>'; });
        h += '</div>';
      }

      if (d.arbiter_edits && d.arbiter_edits.length > 0) {
        h += '<div class="edit-list">';
        d.arbiter_edits.forEach(e => {
          h += '<div class="edit-item"><div class="edit-action ' + editActionCls(e.action) + '">' + esc(e.action) + '</div>';
          h += '<div class="edit-target">' + esc(e.target) + '</div>';
          if (e.replacement) h += '<div class="edit-repl">&rarr; ' + esc(e.replacement) + '</div>';
          h += '</div>';
        });
        h += '</div>';
      }

      if (d.arbiter_policy_notes && d.arbiter_policy_notes.length > 0) {
        h += '<div class="policy-notes">';
        d.arbiter_policy_notes.forEach(n => { h += '<div>' + esc(n) + '</div>'; });
        h += '</div>';
      }
      h += '</div>';
    }

    // Rewrite + re-verify
    if (d.rewrite_occurred) {
      h += '<div class="step-card"><div class="step-label rw">Rewrite &amp; Re-verify &mdash; ' + esc(d.rewrite_verdict) + '</div>';
      h += '<div class="step-content">' + esc(d.rewrite_output) + '</div>';
      h += renderClaimTable(d.rewrite_claim_table);
      h += renderViolations(d.rewrite_violations);
      h += '</div>';
    }
  }

  // Prompt flags
  if (d.prompt_flags) {
    const flags = Object.entries(d.prompt_flags).filter(([k,v]) => v).map(([k]) => k).join(', ');
    if (flags) h += '<div style="font-size:11px;color:#444;margin-top:8px;">Flags: ' + esc(flags) + '</div>';
  }

  h += '</div>'; // details-body
  h += '</div>'; // result-body

  card.innerHTML = h;
  container.appendChild(card);
}

// ---- Stress test ----
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
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const txt = await resp.text();
      let msg = 'HTTP ' + resp.status;
      try { msg += ': ' + JSON.parse(txt).detail; } catch(_) { msg += ': ' + txt.slice(0, 200); }
      log.innerHTML += '<span class="fail">ERROR: ' + esc(msg) + '</span>';
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
        try { d = JSON.parse(line); } catch(_) { log.innerHTML += '<span class="fail">' + esc(line) + '</span>\\n'; continue; }
        if (d.type === 'progress') {
          let cls = d.verdict === 'PASS' ? 'pass' : 'fail';
          let extra = '';
          if (d.arbiter) extra += ' <span class="arb">arbiter:' + esc(d.arbiter) + '</span>';
          if (d.rewrite) extra += ' <span class="rew">[rewrite]</span>';
          log.innerHTML += '[' + d.index + '/' + d.total + '] ' + esc(d.id) + ' <span class="' + cls + '">' + d.verdict + '</span>' + extra + ' (' + d.duration_s + 's)\\n';
          log.scrollTop = log.scrollHeight;
        } else if (d.type === 'summary') {
          renderStressSummary(d, scoreDiv);
        } else if (d.type === 'error') {
          log.innerHTML += '<span class="fail">ERROR: ' + esc(d.error || 'Unknown error') + '</span>\\n';
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
  h += '<div class="pss-summary">Tests: ' + d.total_tests + ' | PASS: ' + d.total_pass + ' | FAIL: ' + d.total_fail + ' | Errors: ' + d.total_error + ' | Avg: ' + d.avg_duration_s + 's</div>';

  // Metrics cards (now includes HAR)
  const m = pss.metrics;
  const p = pss.penalties;
  const cards = [
    {label: 'HBR', desc: 'Halluc. Block Rate', val: ((m.HBR||0)*100).toFixed(1) + '%', pen: p.P1},
    {label: 'HAR', desc: 'Halluc. Avoidance', val: ((m.HAR||0)*100).toFixed(1) + '%', pen: null},
    {label: 'FPF', desc: 'False-Positive FAIL', val: ((m.FPF||0)*100).toFixed(1) + '%', pen: p.P2},
    {label: 'MCP', desc: 'Min Compliance', val: ((m.MCP||0)*100).toFixed(1) + '%', pen: p.P3},
    {label: 'RLS', desc: 'Rewrite Loops', val: (m.RLS||0).toFixed(2), pen: p.P4},
    {label: 'EOI', desc: 'Overreach', val: ((m.EOI||0)*100).toFixed(1) + '%', pen: p.P5},
  ];
  h += '<div class="metrics-grid">';
  cards.forEach(c => {
    h += '<div class="metric-card"><div class="mv">' + c.val + '</div><div class="ml">' + c.label + '</div><div class="ml">' + c.desc + '</div>';
    if (c.pen !== null) h += '<div class="mp">-' + c.pen.toFixed(1) + '</div>';
    h += '</div>';
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
    h += '<div style="margin-top:14px;font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.5px;">Top Violation Reasons</div>';
    for (const v in viols) {
      h += '<div style="font-size:12px;color:#ff453a;margin:3px 0;">' + esc(v) + ': ' + viols[v] + '</div>';
    }
  }

  el.innerHTML = h;
}

// ---- Init ----
loadConfig();
document.getElementById('ui').focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
