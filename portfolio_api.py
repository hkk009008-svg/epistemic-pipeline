from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
import openai
from tavily import TavilyClient

APP_VERSION = "1.3.0"

app = FastAPI(title="GPT-1 > GPT-2 > GPT-3 Verification Pipeline", version=APP_VERSION)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return JSON error envelope for all unhandled exceptions."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": True, "detail": exc.detail},
        )
    return JSONResponse(
        status_code=500,
        content={"error": True, "detail": f"Internal server error: {str(exc)[:200]}"},
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

class TavilyConfig(BaseModel):
    api_key: str
    enabled: bool = True

class SearchSource(BaseModel):
    title: str
    url: str
    snippet: str
    score: float = 0.0

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
    gpt1_output_sanitized: str = ""
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
    # Web search enrichment
    search_performed: bool = False
    search_query: str = ""
    search_sources: List[SearchSource] = []

# NOTE: Global mutable dict shared across all requests. Acceptable for a
# single-user portfolio/demo app, but would need per-session isolation
# (e.g. a session store keyed by cookie/token) in a multi-user deployment.
_openai_config: dict = {}
_openai_client: openai.OpenAI | None = None  # cached client, recreated when config changes
_openai_client_key: str = ""  # tracks which api_key the cached client was built with

_tavily_config: dict = {}  # {"api_key": "tvly-...", "enabled": True}
_tavily_client: TavilyClient | None = None
_tavily_client_key: str = ""

MAX_REWRITE_LOOPS = 3  # max rewrite iterations before giving up

# Activation phrases that bypass GPT-2 verification
ACTIVATION_PATTERNS = [
    r"^Production active\.$",
    r"^Audit v\d+$",
    r"^System initialized$",
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
    'IMPORTANT keyword-matching rules (to reduce false positives):\n'
    '- Match trigger keywords as STANDALONE WORDS only. Do NOT treat substrings as matches.\n'
    '  "dataset", "database", "metadata" do NOT contain the keyword "data".\n'
    '  "researcher" does NOT trigger "research" unless used without a citation.\n'
    '- The exact phrase "Unknown (Actionable): No authoritative dataset available." is a REQUIRED\n'
    '  abstention output. It MUST NOT trigger any finding by itself.\n'
    '- The phrase "Unknown (Actionable)" or "Unknown (Structural)" MUST NOT be treated as unsupported.\n\n'
    '  HARD severity (additional):\n'
    '    - "Unverified statistic" — GPT-1 provides a specific percentage/rate/number when asked,\n'
    '      but without citing an authoritative dataset in the same claim\n\n'
    'Rules:\n'
    '- "studies/data/research/generally/often/suggests" as standalone words without citation+numeric/quotable support -> finding type "Unsupported evidence reference", severity "soft"\n'
    '- Unverified statistic rule: if ORIGINAL PROMPT asks for a percentage/rate AND GPT-1 provides a specific\n'
    '  quantitative figure (e.g., "30%", "one in ten") without citing an authoritative dataset -> "Unverified statistic", severity "hard"\n'
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
    '0) If ALL GPT-2 violations are soft-severity only (Prescriptive creep, '
    'Unsupported evidence reference, Missing jurisdiction, Overconfidence):\n'
    '   -> arbiter_decision = "ALLOW_WITH_EDITS" (NEVER "BLOCK" for soft-only violations)\n'
    '   -> edits_for_gpt1 must DELETE or REWRITE the exact offending phrases.\n\n'
    '1) If GPT-1 contains any fabricated statistic or specific legal conclusion without verification:\n'
    '   -> Prefer "ALLOW_WITH_EDITS" if the offending fragment can be deleted or moved to Unknown.\n'
    '   -> "BLOCK" ONLY if the output cannot be repaired without rewriting the entire substance.\n\n'
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
# Case-sensitive abbreviations (US, UK, EU) — must NOT be folded into (?i)
_JURISDICTION_ABBREV_RE = re.compile(r"\b(?:US|UK|EU)\b")
# Case-insensitive jurisdiction names
_JURISDICTION_RE = re.compile(
    r"(?i)\b(?:federal|state of"
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
        "jurisdiction_present": bool(_JURISDICTION_ABBREV_RE.search(prompt) or _JURISDICTION_RE.search(prompt)),
        "future_year": bool(_FUTURE_YEAR_RE.search(prompt)),
    }


# ---- Deterministic Sanitizer ----
_BANNED_EVIDENCE_WORDS = [
    "studies", "research", "data", "generally", "often",
    "suggests", "typically", "usually", "commonly",
]
_BANNED_EVIDENCE_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _BANNED_EVIDENCE_WORDS)) + r")\b"
    r"(?!\s*\([^)]+\))"   # not followed by a parenthetical citation
    r"(?!\s*\[[^\]]+\])",  # not followed by a bracket citation
    re.IGNORECASE,
)
_BANNED_SUBS = {
    "studies": "published material",
    "research": "published material",
    "data": "evidence",
    "generally": "in some cases",
    "often": "in some cases",
    "suggests": "may indicate",
    "typically": "in some cases",
    "usually": "in some cases",
    "commonly": "in some cases",
}
_BARE_PERCENT_RE = re.compile(
    r"(?i)\b(?:about|roughly|approximately|around|nearly|close to|an estimated|estimated)?\s*"
    r"\d+(?:\.\d+)?\s*(?:%|percent)\b"
)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially)\b"
)
_OUTCOME_KW = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase|improve your|"
    r"could help|could assist|may improve|may help|could potentially|"
    r"guarantee|ensure|succeed)\b"
)
_PRO_MENTION_RE = re.compile(
    r"(?i)\b(?:hire|hiring|attorney|lawyer|broker|specialist|accountant|consultant)\b"
)
_PRO_BENEFIT_RE = re.compile(
    r"(?i)\b(?:help|helps|assist|assists|improve|improves|reduce|reduces"
    r"|increase|increases|significantly|better|boost)\b"
)
ROLE_DEFINITION_STMT = (
    "An attorney's function is to advise on legal requirements and prepare filings; "
    "whether that changes outcomes depends on case-specific factors and cannot be guaranteed."
)
_UNKNOWN_ACTIONABLE_LINE = "Unknown (Actionable): No authoritative dataset available."


def sanitize_output(text: str, flags: dict) -> str:
    """Pre-clean GPT-1 output deterministically before GPT-2 verification.

    Uses word-for-word substitution (not deletion) to preserve grammar.
    """
    if not text:
        return ""
    out = text

    # 1. Substitute banned evidence words (word-boundary, not delete)
    def _sub_word(m):
        return _BANNED_SUBS.get(m.group(0).lower(), m.group(0))
    out = _BANNED_EVIDENCE_RE.sub(_sub_word, out)

    # 2. If percent_requested: strip sentences with bare stats, ensure Unknown(Actionable)
    if flags.get("percent_requested"):
        sentences = re.split(r"(?<=[.!?])\s+", out)
        kept = [s for s in sentences if not _BARE_PERCENT_RE.search(s)]
        out = " ".join(kept)
        if _UNKNOWN_ACTIONABLE_LINE not in out:
            out += "\n\n" + _UNKNOWN_ACTIONABLE_LINE
    else:
        # For non-percent requests, still replace bare % with Unknown marker
        out = _BARE_PERCENT_RE.sub(
            "Unknown (Actionable): No authoritative dataset available for this figure", out
        )

    # 3. Strip outcome-promise phrases
    out = _OUTCOME_PROMISE_RE.sub("", out)

    # 4. Professional mention: remove benefit-framing sentences, enforce role definition
    if flags.get("advice_requested") or _PRO_MENTION_RE.search(text):
        sentences = re.split(r"(?<=[.!?])\s+", out)
        cleaned = [s for s in sentences if not (_PRO_MENTION_RE.search(s) and _PRO_BENEFIT_RE.search(s))]
        out = " ".join(cleaned)
        if _PRO_MENTION_RE.search(text) and ROLE_DEFINITION_STMT not in out:
            out += "\n\n" + ROLE_DEFINITION_STMT

    # 5. If advice NOT requested, strip "Options" section
    if not flags.get("advice_requested"):
        out = re.sub(
            r"\n\s*(?:Options|Recommendations)\s*\n[\s\S]*?(?=\n\s*\d\)\s|\Z)",
            "\n", out, flags=re.IGNORECASE,
        )

    # 6. Normalize whitespace
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" +\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


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

    # If it starts with prose before JSON, extract the JSON object.
    # Find the first '{' and the last '}' to correctly handle nested braces.
    if not cleaned.startswith("{"):
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last > first:
            cleaned = cleaned[first:last + 1]

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

        # Deterministic severity map — override GPT-2's classification to
        # prevent the LLM from mis-labeling soft violations as hard.
        _HARD_TYPES = frozenset({
            "Fabricated statistic",
            "Fabricated citation",
            "False legal conclusion",
            "Unverified statistic",
        })

        findings = []  # type: List[dict]
        for f in raw_findings:
            ftype = f.get("type", "")
            # Enforce severity by finding type — GPT-2 sometimes assigns
            # "hard" to types that should always be "soft" (and vice versa).
            severity = "hard" if ftype in _HARD_TYPES else "soft"
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

            # Context-aware filter: if percent was requested and GPT-1 correctly
            # abstained (Unknown Actionable), suppress findings that flag
            # the standard abstention phrasing (e.g., "dataset").
            if (
                flags
                and flags.get("percent_requested")
                and ftype == "Unsupported evidence reference"
                and severity == "soft"
                and ("dataset" in detail.lower() or "unknown" in detail.lower())
            ):
                continue

            # Context-aware filter: if jurisdiction IS present in the prompt,
            # suppress "Missing jurisdiction" findings.
            if (
                flags
                and flags.get("jurisdiction_present")
                and ftype == "Missing jurisdiction"
                and severity == "soft"
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


def _get_openai_client() -> openai.OpenAI:
    """Return a cached OpenAI client, recreating only when the API key changes."""
    global _openai_client, _openai_client_key
    current_key = _openai_config.get("api_key", "")
    if _openai_client is None or _openai_client_key != current_key:
        _openai_client = openai.OpenAI(api_key=current_key)
        _openai_client_key = current_key
    return _openai_client


def _get_tavily_client() -> TavilyClient | None:
    """Return a cached TavilyClient, or None if not configured/disabled."""
    global _tavily_client, _tavily_client_key
    if not _tavily_config.get("enabled", False):
        return None
    current_key = _tavily_config.get("api_key", "")
    if not current_key:
        return None
    if _tavily_client is None or _tavily_client_key != current_key:
        _tavily_client = TavilyClient(api_key=current_key)
        _tavily_client_key = current_key
    return _tavily_client


def perform_web_search(query: str, max_results: int = 5) -> tuple[list[SearchSource], str]:
    """Call Tavily search API. Returns (sources, raw_context_string).

    Returns empty results on any error (search is best-effort).
    """
    client = _get_tavily_client()
    if client is None:
        return [], ""

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
            topic="general",
        )
    except Exception:
        return [], ""

    sources = []
    for r in response.get("results", []):
        sources.append(SearchSource(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            score=r.get("score", 0.0),
        ))

    context_lines = []
    for i, s in enumerate(sources, 1):
        context_lines.append(f"[{i}] {s.title}\n    URL: {s.url}\n    Excerpt: {s.snippet}")

    raw_context = "\n\n".join(context_lines)

    answer = response.get("answer", "")
    if answer:
        raw_context = f"Tavily Summary: {answer}\n\n---\nSources:\n{raw_context}"

    return sources, raw_context


def _should_search(flags: dict) -> bool:
    """Determine if web search should be triggered based on prompt routing flags."""
    if not _tavily_config.get("enabled", False):
        return False
    if not _tavily_config.get("api_key", ""):
        return False
    return (
        flags.get("percent_requested", False)
        or flags.get("legal_mode", False)
        or flags.get("future_year", False)
    )


def _build_gpt2_prompt(original_prompt: str, text_to_verify: str) -> str:
    """Build the user prompt sent to GPT-2 for verification."""
    return f"ORIGINAL PROMPT:\n{original_prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{text_to_verify}"


def _build_response(
    *,
    gpt1_input: str,
    gpt1_output: str,
    gpt1_output_sanitized: str = "",
    bypassed: bool = False,
    gpt2_raw: str = "",
    claim_table: list | None = None,
    violations: list | None = None,
    gpt2_verdict: str = "",
    arbiter_invoked: bool = False,
    arbiter_decision: str = "",
    arbiter_rationale: list | None = None,
    arbiter_edits: list | None = None,
    arbiter_policy_notes: list | None = None,
    arbiter_raw: str = "",
    rewrite_occurred: bool = False,
    rewrite_output: str = "",
    rewrite_gpt2_raw: str = "",
    rewrite_claim_table: list | None = None,
    rewrite_violations: list | None = None,
    rewrite_verdict: str = "",
    final_verdict: str = "",
    final_result: str = "",
    prompt_flags: dict | None = None,
    sanitizer_applied: bool = False,
    search_performed: bool = False,
    search_query: str = "",
    search_sources: list | None = None,
) -> PipelineResponse:
    """Construct a PipelineResponse with sensible defaults for optional fields."""
    return PipelineResponse(
        gpt1_input=gpt1_input,
        gpt1_output=gpt1_output,
        gpt1_output_sanitized=gpt1_output_sanitized,
        bypassed=bypassed,
        gpt2_raw=gpt2_raw,
        claim_table=claim_table or [],
        violations=violations or [],
        gpt2_verdict=gpt2_verdict,
        arbiter_invoked=arbiter_invoked,
        arbiter_decision=arbiter_decision,
        arbiter_rationale=arbiter_rationale or [],
        arbiter_edits=arbiter_edits or [],
        arbiter_policy_notes=arbiter_policy_notes or [],
        arbiter_raw=arbiter_raw,
        rewrite_occurred=rewrite_occurred,
        rewrite_output=rewrite_output,
        rewrite_gpt2_raw=rewrite_gpt2_raw,
        rewrite_claim_table=rewrite_claim_table or [],
        rewrite_violations=rewrite_violations or [],
        rewrite_verdict=rewrite_verdict,
        final_verdict=final_verdict,
        final_result=final_result,
        prompt_flags=prompt_flags,
        sanitizer_applied=sanitizer_applied,
        search_performed=search_performed,
        search_query=search_query,
        search_sources=search_sources or [],
    )


# ---- Module-level cache for tests.json ----
_tests_cache: list | None = None
_tests_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests.json")


def _load_tests() -> list:
    """Load tests.json once and cache at module level."""
    global _tests_cache
    if _tests_cache is None:
        if not os.path.exists(_tests_path):
            raise HTTPException(status_code=404, detail="tests.json not found next to portfolio_api.py")
        with open(_tests_path, "r", encoding="utf-8") as f:
            _tests_cache = json.load(f)
    return _tests_cache


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
    return {"status": "ok", "model": config.model, "key_set": True}

@app.get("/api/openai/config")
def get_openai_config():
    if "api_key" not in _openai_config:
        return {"key_set": False}
    masked = _openai_config["api_key"][:8] + "..." + _openai_config["api_key"][-4:]
    return {"key_set": True, "model": _openai_config.get("model"), "key_preview": masked}


@app.post("/api/tavily/config")
def set_tavily_config(config: TavilyConfig):
    clean_key = config.api_key.strip()
    clean_key = clean_key.encode("ascii", errors="ignore").decode("ascii")
    clean_key = clean_key.replace(" ", "")
    if not clean_key:
        raise HTTPException(status_code=400, detail="Invalid Tavily API key.")
    _tavily_config["api_key"] = clean_key
    _tavily_config["enabled"] = config.enabled
    return {"status": "ok", "enabled": config.enabled, "key_set": True}


@app.get("/api/tavily/config")
def get_tavily_config():
    if "api_key" not in _tavily_config:
        return {"key_set": False, "enabled": False}
    masked = _tavily_config["api_key"][:8] + "..." + _tavily_config["api_key"][-4:]
    return {
        "key_set": True,
        "enabled": _tavily_config.get("enabled", False),
        "key_preview": masked,
    }


@app.post("/api/tavily/toggle")
def toggle_tavily(enabled: bool = True):
    if "api_key" not in _tavily_config:
        raise HTTPException(status_code=400, detail="Set Tavily API key first.")
    _tavily_config["enabled"] = enabled
    return {"status": "ok", "enabled": enabled}


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


@app.post("/api/pipeline", response_model=PipelineResponse)
def run_pipeline(req: PipelineRequest):
    if "api_key" not in _openai_config:
        raise HTTPException(status_code=400, detail="Set your OpenAI API key first.")

    client = _get_openai_client()
    model = _openai_config.get("model", "gpt-4o-mini")

    # ---- Deterministic prompt routing ----
    flags = route_prompt(req.prompt)

    gpt1_system = req.gpt1_system or DEFAULT_GPT1_SYSTEM
    gpt2_system = req.gpt2_system or DEFAULT_GPT2_SYSTEM
    gpt3_system = req.gpt3_system or DEFAULT_GPT3_SYSTEM

    # ---- Flag-driven GPT-1 system prompt augmentation ----
    if flags.get("advice_requested"):
        gpt1_system += (
            "\n\nNOTE: The user is explicitly requesting advice/options. "
            "You may provide conditional process guidance."
        )
    if flags.get("percent_requested"):
        gpt1_system += (
            "\n\nIMPORTANT: The user is asking for a percentage/rate/statistic. "
            "You MUST NOT fabricate any number. If no authoritative dataset is available, "
            "output exactly: Unknown (Actionable): No authoritative dataset available."
        )
    if flags.get("legal_mode") and not flags.get("jurisdiction_present"):
        gpt1_system += (
            "\n\nIMPORTANT: The user asks a legal/regulatory question but has NOT specified "
            "a jurisdiction. You MUST NOT conclude legality/illegality. State in Unknowns: "
            "Unknown (Actionable): Jurisdiction not specified; conclusion depends on applicable law."
        )
    if flags.get("future_year"):
        gpt1_system += (
            "\n\nIMPORTANT: The question references a future year. Do NOT assume current law "
            "or policy will persist. Frame all forward-looking claims as Unknown (Structural)."
        )

    # ---- Web Search Enrichment (before GPT-1) ----
    search_sources: list[SearchSource] = []
    search_context = ""
    search_performed = False

    if _should_search(flags):
        search_sources, search_context = perform_web_search(req.prompt)
        search_performed = len(search_sources) > 0

    gpt1_user_content = req.prompt
    if search_performed and search_context:
        gpt1_system += (
            "\n\nWEB SEARCH RESULTS are provided below the user's question. "
            "You MUST ground your response in these sources. "
            "When citing a fact from a source, reference it as [1], [2], etc. "
            "If a source provides a specific statistic, you may quote it with the citation. "
            "Do NOT fabricate additional sources beyond what is provided. "
            "If the sources do not contain the answer, state Unknown (Actionable)."
        )
        gpt1_user_content = (
            f"{req.prompt}\n\n"
            f"--- WEB SEARCH RESULTS ---\n"
            f"{search_context}\n"
            f"--- END SEARCH RESULTS ---"
        )

        # Augment GPT-2 to recognize the provided sources
        source_summary = "; ".join(
            f'[{i}] "{s.title}" ({s.url})' for i, s in enumerate(search_sources, 1)
        )
        gpt2_system += (
            "\n\nIMPORTANT: GPT-1 was given web search results from the following sources:\n"
            f"{source_summary}\n\n"
            "When evaluating claims:\n"
            "- If a claim cites one of these sources (e.g., [1], [2]) and the source snippet "
            "supports the claim, categorize it as 'Supported' (not 'Unsupported').\n"
            "- A statistic that is attributed to a provided source is NOT 'Fabricated' or "
            "'Unverified' -- categorize it as 'Supported'.\n"
            "- If GPT-1 cites a source number that does not exist in the list above, "
            "flag it as 'Fabricated citation'.\n"
            "- Claims NOT backed by any provided source should still be evaluated normally."
        )

    # ---- Step 1: GPT-1 Generate ----
    gpt1_output = call_openai(client, model, gpt1_system, gpt1_user_content)

    # ---- Activation bypass ----
    if is_activation_phrase(gpt1_output):
        return _build_response(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=True,
            gpt2_raw="(bypassed)", gpt2_verdict="PASS",
            final_verdict="PASS", final_result=gpt1_output,
            prompt_flags=flags,
            search_performed=search_performed,
            search_query=req.prompt if search_performed else "",
            search_sources=search_sources,
        )

    # ---- Deterministic sanitizer (pre-clean before GPT-2) ----
    sanitized_output = sanitize_output(gpt1_output, flags)
    sanitizer_applied = (sanitized_output != gpt1_output)

    # ---- Step 2: GPT-2 Verify (on sanitized output) ----
    gpt2_raw = call_openai(client, model, gpt2_system,
                           _build_gpt2_prompt(req.prompt, sanitized_output),
                           expect_json=True)
    claim_table, violations, gpt2_verdict, findings = parse_gpt2(gpt2_raw, flags=flags)

    # ---- If GPT-2 PASS: done ----
    if gpt2_verdict == "PASS":
        return _build_response(
            gpt1_input=req.prompt, gpt1_output=gpt1_output,
            gpt1_output_sanitized=sanitized_output,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS",
            final_verdict="PASS", final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            search_performed=search_performed,
            search_query=req.prompt if search_performed else "",
            search_sources=search_sources,
        )

    # ---- GPT-2 FAIL: soft-only → auto-pass (sanitizer already cleaned) ----
    if _all_soft(findings):
        # All violations are soft and the deterministic sanitizer has already
        # been applied. A second sanitize_output call on already-sanitized text
        # is a no-op, so skip it and pass with a note instead.
        return _build_response(
            gpt1_input=req.prompt, gpt1_output=gpt1_output,
            gpt1_output_sanitized=sanitized_output,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict,
            final_verdict="PASS (soft-only, sanitizer applied)",
            final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            search_performed=search_performed,
            search_query=req.prompt if search_performed else "",
            search_sources=search_sources,
        )

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

    # Override: if all GPT-2 findings are soft, never BLOCK — force ALLOW_WITH_EDITS
    if arbiter_decision == "BLOCK" and _all_soft(findings):
        arbiter_decision = "ALLOW_WITH_EDITS"
        arbiter_rationale.append(
            "Overridden: all violations are soft-severity; "
            "defaulting to ALLOW_WITH_EDITS per adjudication policy 0."
        )

    # ---- Decision: BLOCK ----
    if arbiter_decision == "BLOCK":
        return _build_response(
            gpt1_input=req.prompt, gpt1_output=gpt1_output,
            gpt1_output_sanitized=sanitized_output,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict,
            arbiter_invoked=True, arbiter_decision="BLOCK",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            final_verdict="FAIL", final_result="NO PASS",
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            search_performed=search_performed,
            search_query=req.prompt if search_performed else "",
            search_sources=search_sources,
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
        re_gpt2_raw = call_openai(client, model, gpt2_system,
                                  _build_gpt2_prompt(req.prompt, rewrite_output),
                                  expect_json=True)
        re_ct, re_viol, re_verdict, _ = parse_gpt2(re_gpt2_raw, flags=flags)

        return _build_response(
            gpt1_input=req.prompt, gpt1_output=gpt1_output,
            gpt1_output_sanitized=sanitized_output,
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
            search_performed=search_performed,
            search_query=req.prompt if search_performed else "",
            search_sources=search_sources,
        )

    # ---- Decision: ALLOW_WITH_EDITS (with rewrite loop) ----
    rewrite_output = sanitized_output
    re_gpt2_raw = ""
    re_ct: list = []
    re_viol: list = []
    re_verdict = "FAIL"
    re_findings: list = []

    for _loop in range(MAX_REWRITE_LOOPS):
        rewrite_prompt = apply_edits(rewrite_output, arbiter_edits)
        rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

        # Re-verify the rewritten output with GPT-2
        re_gpt2_raw = call_openai(client, model, gpt2_system,
                                  _build_gpt2_prompt(req.prompt, rewrite_output),
                                  expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)
        if re_verdict == "PASS":
            break

    # If still failing on soft-only violations after rewrite loop, force Unknown-only
    # and actually re-verify to avoid marking unverified text as PASS.
    if re_verdict == "FAIL" and _all_soft(re_findings):
        unknown_prompt = (
            f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
            f"Remaining soft violations could not be resolved. "
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, unknown_prompt)
        # Re-verify the unknown-only rewrite with GPT-2
        re_gpt2_raw = call_openai(client, model, gpt2_system,
                                  _build_gpt2_prompt(req.prompt, rewrite_output),
                                  expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)
        if re_verdict != "PASS":
            re_verdict = "PASS (unverified)"

    return _build_response(
        gpt1_input=req.prompt, gpt1_output=gpt1_output,
        gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
        gpt2_verdict=gpt2_verdict,
        arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
        arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
        arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
        rewrite_occurred=True, rewrite_output=rewrite_output,
        rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
        rewrite_violations=re_viol, rewrite_verdict=re_verdict,
        final_verdict=re_verdict,
        final_result=rewrite_output if "PASS" in re_verdict else "NO PASS",
        prompt_flags=flags, sanitizer_applied=sanitizer_applied,
        search_performed=search_performed,
        search_query=req.prompt if search_performed else "",
        search_sources=search_sources,
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

    # HPR — Hallucination Prevention Rate
    # (blocked OR safely abstained with "Unknown (Actionable)") / fabrication_attempts
    if fab_attempts:
        safely_handled = [
            r for r in fab_attempts
            if r["final_verdict"] == "FAIL"
            or "Unknown (Actionable)" in r.get("final_result", "")
        ]
        HPR = len(safely_handled) / len(fab_attempts)
    else:
        HPR = 1.0

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
        "metrics": {"HLR": round(HLR, 4), "HPR": round(HPR, 4), "FPF": round(FPF, 4), "MCP": round(MCP, 4), "RLS": round(RLS, 4), "EOI": round(EOI, 4)},
        "penalties": {"P1": round(P1, 2), "P2": round(P2, 2), "P3": round(P3, 2), "P4": round(P4, 2), "P5": round(P5, 2)},
    }


@app.get("/api/stress/categories")
def get_stress_categories():
    """Return list of unique categories from tests.json."""
    tests = _load_tests()
    cats = sorted(set(t["category"] for t in tests))
    return cats


@app.post("/api/stress")
def run_stress_test(req: StressRequest):
    """Run stress harness inline — returns streaming JSONL progress + final score."""
    if "api_key" not in _openai_config:
        raise HTTPException(status_code=400, detail="Set your OpenAI API key first.")

    # Load tests (cached at module level)
    all_tests = _load_tests()
    tests = list(all_tests)  # shallow copy so filtering doesn't mutate cache

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
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Epistemic Pipeline OS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #e8edf5;
    --bg-soft: #f4f7fb;
    --ink: #163046;
    --ink-soft: #44617a;
    --line: rgba(25, 58, 86, 0.14);
    --panel: rgba(255, 255, 255, 0.76);
    --panel-strong: rgba(255, 255, 255, 0.9);
    --accent: #1677c5;
    --accent-2: #0f9f8f;
    --danger: #b13c3c;
    --warn: #ca7a15;
    --ok: #1f8b4c;
    --radius-lg: 18px;
    --radius-md: 12px;
    --shadow: 0 14px 40px rgba(20, 52, 80, 0.16);
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    height: 100%;
    color: var(--ink);
    font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 500px at 18% -10%, #d7eef8 0%, transparent 60%),
                radial-gradient(900px 600px at 100% 100%, #d8f6ec 0%, transparent 55%),
                linear-gradient(160deg, #eaf0f8, #dfe8f3);
  }

  .ambient {
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: linear-gradient(transparent 96%, rgba(255,255,255,0.23) 97%),
                      linear-gradient(90deg, transparent 96%, rgba(255,255,255,0.2) 97%);
    background-size: 26px 26px;
    opacity: 0.35;
    z-index: 0;
  }

  .menubar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 38px;
    z-index: 30;
    background: rgba(243, 247, 252, 0.82);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(25, 58, 86, 0.14);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    font-size: 13px;
  }

  .menu-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
  }

  .menu-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1e88d6, #0f9f8f);
    box-shadow: 0 0 0 3px rgba(22, 119, 197, 0.14);
  }

  .window {
    position: relative;
    z-index: 2;
    width: min(1360px, calc(100vw - 28px));
    margin: 56px auto 14px;
    height: calc(100vh - 70px);
    border-radius: 20px;
    border: 1px solid var(--line);
    background: var(--panel);
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    animation: window-in 360ms ease;
  }

  @keyframes window-in {
    from { opacity: 0; transform: translateY(8px) scale(0.995); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }

  .window-head {
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.35));
  }

  .lights {
    display: flex;
    gap: 8px;
  }

  .light {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1px solid rgba(0,0,0,0.14);
  }

  .light.red { background: #ff6f64; }
  .light.yellow { background: #f5bf4f; }
  .light.green { background: #4dc55a; }

  .title {
    font-size: 13px;
    font-weight: 600;
    color: var(--ink-soft);
    letter-spacing: 0.3px;
  }

  .head-actions {
    display: flex;
    gap: 8px;
  }

  button,
  input,
  select,
  textarea {
    font: inherit;
    color: inherit;
  }

  .btn {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--panel-strong);
    color: var(--ink);
    height: 34px;
    padding: 0 14px;
    cursor: pointer;
    font-weight: 600;
    transition: all 140ms ease;
  }

  .btn:hover { transform: translateY(-1px); }

  .btn.primary {
    color: #fff;
    border: none;
    background: linear-gradient(135deg, var(--accent), #0a66ad);
    box-shadow: 0 7px 16px rgba(22, 119, 197, 0.28);
  }

  .btn.ghost {
    background: transparent;
  }

  .btn.warn {
    border-color: rgba(177, 60, 60, 0.4);
    color: var(--danger);
  }

  .window-body {
    flex: 1;
    display: grid;
    grid-template-columns: 220px 1fr 300px;
    min-height: 0;
  }

  .sidebar {
    border-right: 1px solid var(--line);
    background: rgba(250, 252, 255, 0.64);
    padding: 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .nav-item {
    border: 1px solid transparent;
    border-radius: 12px;
    background: transparent;
    color: var(--ink-soft);
    font-weight: 600;
    text-align: left;
    padding: 10px 12px;
    cursor: pointer;
    transition: all 140ms ease;
  }

  .nav-item.active {
    color: var(--accent);
    border-color: rgba(22, 119, 197, 0.22);
    background: rgba(22, 119, 197, 0.09);
  }

  .side-note {
    margin-top: auto;
    font-size: 12px;
    color: var(--ink-soft);
    line-height: 1.4;
    padding: 10px;
    border-radius: 12px;
    border: 1px dashed rgba(22, 119, 197, 0.28);
    background: rgba(22, 119, 197, 0.05);
  }

  .workspace {
    min-width: 0;
    padding: 16px;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .view {
    display: none;
    flex: 1;
    min-height: 0;
    overflow: auto;
    animation: fade-in 160ms ease;
  }

  .view.active {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  @keyframes fade-in {
    from { opacity: 0; transform: translateY(3px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .card {
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 14px;
    background: rgba(255,255,255,0.66);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
  }

  .hero h2 {
    margin: 0;
    font-size: 19px;
    font-weight: 700;
  }

  .hero p {
    margin: 8px 0 0;
    color: var(--ink-soft);
    line-height: 1.45;
    font-size: 14px;
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 160px;
    padding-right: 2px;
  }

  .stage {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: rgba(255, 255, 255, 0.8);
    overflow: hidden;
  }

  .stage-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 12px;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
    background: rgba(252, 254, 255, 0.8);
  }

  .stage-head h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 700;
  }

  .stage-body {
    padding: 12px;
    font-size: 13px;
    line-height: 1.5;
  }

  .stage pre {
    margin: 0;
    white-space: pre-wrap;
    font-family: "IBM Plex Mono", Menlo, monospace;
    font-size: 12px;
    color: #224760;
  }

  .tone-user .stage-head { background: rgba(22, 119, 197, 0.09); }
  .tone-generator .stage-head { background: rgba(15, 159, 143, 0.1); }
  .tone-verifier .stage-head { background: rgba(202, 122, 21, 0.1); }
  .tone-arbiter .stage-head { background: rgba(67, 119, 169, 0.1); }
  .tone-final-ok .stage-head { background: rgba(31, 139, 76, 0.12); }
  .tone-final-fail .stage-head { background: rgba(177, 60, 60, 0.12); }
  .tone-error .stage-head { background: rgba(177, 60, 60, 0.12); }
  .tone-search .stage-head { background: rgba(106, 76, 219, 0.1); }

  .pill {
    border-radius: 999px;
    font-size: 11px;
    padding: 3px 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }

  .pill.ok { color: var(--ok); background: rgba(31, 139, 76, 0.14); }
  .pill.warn { color: var(--warn); background: rgba(202, 122, 21, 0.14); }
  .pill.bad { color: var(--danger); background: rgba(177, 60, 60, 0.13); }
  .pill.neutral { color: var(--ink-soft); background: rgba(68, 97, 122, 0.13); }

  .mono {
    font-family: "IBM Plex Mono", Menlo, monospace;
    font-size: 12px;
  }

  .table-wrap {
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255,255,255,0.7);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  th, td {
    border-bottom: 1px solid rgba(25, 58, 86, 0.1);
    padding: 8px;
    text-align: left;
    vertical-align: top;
  }

  th {
    font-size: 11px;
    letter-spacing: 0.35px;
    text-transform: uppercase;
    color: var(--ink-soft);
  }

  details summary {
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    color: var(--ink-soft);
  }

  .list {
    margin: 8px 0 0;
    padding-left: 18px;
  }

  .list li {
    margin: 4px 0;
  }

  .composer {
    margin-top: auto;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 10px;
    align-items: center;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.9);
    border-radius: 14px;
    padding: 8px;
  }

  .composer input {
    border: none;
    background: transparent;
    height: 40px;
    padding: 0 10px;
    font-size: 15px;
    outline: none;
  }

  .inspector {
    border-left: 1px solid var(--line);
    background: rgba(247, 250, 254, 0.7);
    padding: 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .status-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 7px;
    font-size: 13px;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: var(--ink-soft);
  }

  .status-row strong {
    color: var(--ink);
    font-size: 12px;
  }

  .chip-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }

  .chip {
    border: 1px solid rgba(68, 97, 122, 0.2);
    border-radius: 999px;
    background: rgba(68, 97, 122, 0.08);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 8px;
    color: var(--ink-soft);
  }

  .latest-output {
    min-height: 120px;
    max-height: 220px;
    overflow: auto;
    margin: 0;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.8);
    color: #274b63;
    white-space: pre-wrap;
  }

  .muted {
    color: var(--ink-soft);
    font-size: 12px;
  }

  .settings-grid {
    display: grid;
    gap: 12px;
  }

  label {
    display: grid;
    gap: 6px;
    font-size: 13px;
    color: var(--ink-soft);
  }

  .row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 8px;
  }

  input[type="text"], input[type="password"], input[type="number"], select, textarea {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255,255,255,0.88);
    min-height: 38px;
    padding: 8px 10px;
    outline: none;
  }

  textarea {
    min-height: 96px;
    resize: vertical;
  }

  .stress-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: end;
  }

  .stress-log {
    min-height: 180px;
    max-height: 360px;
    overflow: auto;
    border-radius: 12px;
    border: 1px solid var(--line);
    padding: 10px;
    background: rgba(250,252,255,0.8);
    font-family: "IBM Plex Mono", Menlo, monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #28516b;
  }

  .log-line { margin: 2px 0; }
  .log-line.pass { color: var(--ok); }
  .log-line.fail { color: var(--danger); }
  .log-line.warn { color: var(--warn); }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 8px;
  }

  .summary-box {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px;
    background: rgba(255,255,255,0.8);
  }

  .summary-box .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.45px;
    color: var(--ink-soft);
  }

  .summary-box .value {
    margin-top: 3px;
    font-size: 19px;
    font-weight: 700;
  }

  .loading {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: var(--ink-soft);
  }

  .loading-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(0.8); opacity: 0.55; }
    50% { transform: scale(1); opacity: 1; }
  }

  @media (max-width: 1180px) {
    .window-body { grid-template-columns: 190px 1fr; }
    .inspector { display: none; }
  }

  @media (max-width: 880px) {
    .window {
      width: calc(100vw - 12px);
      margin-top: 44px;
      height: calc(100vh - 52px);
      border-radius: 14px;
    }

    .window-body {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr;
    }

    .sidebar {
      border-right: none;
      border-bottom: 1px solid var(--line);
      flex-direction: row;
      align-items: center;
      overflow-x: auto;
    }

    .side-note { display: none; }
    .nav-item { white-space: nowrap; }
  }
</style>
</head>
<body>
<div class="ambient" aria-hidden="true"></div>

<header class="menubar">
  <div class="menu-left">
    <span class="menu-dot" aria-hidden="true"></span>
    <span>Epistemic OS</span>
  </div>
  <div id="clockText" class="mono" aria-live="polite"></div>
</header>

<section class="window" role="application" aria-label="Epistemic verification workspace">
  <div class="window-head">
    <div class="lights" aria-hidden="true">
      <span class="light red"></span>
      <span class="light yellow"></span>
      <span class="light green"></span>
    </div>
    <div class="title">Generator -> Verifier -> Arbiter Workspace</div>
    <div class="head-actions">
      <button class="btn ghost" id="clearBtn">Clear</button>
      <button class="btn" id="openSettingsBtn">Settings</button>
    </div>
  </div>

  <div class="window-body">
    <aside class="sidebar" aria-label="Navigation">
      <button class="nav-item active" data-view="chatView">Workspace</button>
      <button class="nav-item" data-view="stressView">Diagnostics</button>
      <button class="nav-item" data-view="settingsView">Connection</button>
      <div class="side-note">
        Default behavior is optimized for one action: run pipeline, review verdict, export result.
      </div>
    </aside>

    <section class="workspace">
      <section id="chatView" class="view active" aria-label="Pipeline run view">
        <article class="card hero">
          <h2>Run Verification Pipeline</h2>
          <p>Type a prompt and run once. The interface reveals details progressively: generator output, verifier findings, arbiter decision, and final result.</p>
        </article>

        <section id="timeline" class="timeline" aria-live="polite"></section>

        <form id="promptForm" class="composer">
          <input id="promptInput" type="text" autocomplete="off" placeholder="Ask a question to run through the pipeline...">
          <button id="runBtn" class="btn primary" type="submit">Run Pipeline</button>
        </form>
      </section>

      <section id="stressView" class="view" aria-label="Stress diagnostics view">
        <article class="card">
          <h3 style="margin:0 0 8px;font-size:18px;">Diagnostics: Stress Test</h3>
          <p class="muted" style="margin:0 0 12px;">Runs dataset checks using NDJSON streaming. Use this for regression monitoring, not normal chat flow.</p>
          <div class="stress-controls">
            <label style="min-width:200px;">
              Category
              <select id="stressCategory"></select>
            </label>
            <label style="min-width:140px;">
              Per category
              <input id="stressCount" type="number" min="1" max="20" value="5">
            </label>
            <button id="runStressBtn" class="btn primary" type="button">Run Stress</button>
            <button id="cancelStressBtn" class="btn warn" type="button" disabled>Cancel</button>
          </div>
        </article>
        <article class="card">
          <div class="stress-log" id="stressLog"></div>
          <div id="stressSummary"></div>
        </article>
      </section>

      <section id="settingsView" class="view" aria-label="Configuration view">
        <article class="card settings-grid">
          <h3 style="margin:0;font-size:18px;">Connection</h3>
          <label>
            OpenAI API Key
            <div class="row">
              <input id="apiKeyInput" type="password" placeholder="sk-...">
              <button id="saveConfigBtn" class="btn" type="button">Save</button>
            </div>
          </label>
          <label>
            Model
            <select id="modelSelect">
              <option value="gpt-4o-mini">gpt-4o-mini</option>
              <option value="gpt-4o">gpt-4o</option>
              <option value="gpt-4.1-mini">gpt-4.1-mini</option>
              <option value="gpt-4.1">gpt-4.1</option>
            </select>
          </label>
          <label>
            Access Token (optional)
            <input id="accessTokenInput" type="password" placeholder="Bearer token for PIPELINE_AUTH_TOKEN">
          </label>
          <div id="configStatus" class="muted">Loading configuration...</div>

          <hr style="border:none;border-top:1px solid var(--line);margin:12px 0;">
          <h3 style="margin:0 0 6px;font-size:16px;">Web Search (Tavily)</h3>
          <label>
            Tavily API Key
            <div style="display:flex;gap:6px;">
              <input id="tavilyKeyInput" type="password" placeholder="tvly-..." style="flex:1;">
              <button id="saveTavilyBtn" class="btn" type="button">Save</button>
            </div>
          </label>
          <label style="display:flex;flex-direction:row;align-items:center;gap:10px;cursor:pointer;">
            <input id="tavilyEnabledToggle" type="checkbox" style="width:auto;min-height:auto;">
            Enable web search enrichment
          </label>
          <div id="tavilyStatus" class="muted">Loading Tavily configuration...</div>

          <details>
            <summary>Advanced Prompt Overrides</summary>
            <label>
              GPT-1 System Prompt Override
              <textarea id="g1s" placeholder="Leave empty to use backend default."></textarea>
            </label>
            <label>
              GPT-2 System Prompt Override
              <textarea id="g2s" placeholder="Leave empty to use backend default."></textarea>
            </label>
            <label>
              GPT-3 System Prompt Override
              <textarea id="g3s" placeholder="Leave empty to use backend default."></textarea>
            </label>
          </details>
        </article>
      </section>
    </section>

    <aside class="inspector" aria-label="Run inspector">
      <article class="card">
        <h3 style="margin:0 0 10px;font-size:16px;">Run Status</h3>
        <div class="status-grid">
          <div class="status-row"><span>API Key</span><strong id="keyState">Unknown</strong></div>
          <div class="status-row"><span>Model</span><strong id="modelState">-</strong></div>
          <div class="status-row"><span>Final Verdict</span><strong id="verdictState">-</strong></div>
          <div class="status-row"><span>Arbiter</span><strong id="arbiterState">-</strong></div>
          <div class="status-row"><span>Sanitizer</span><strong id="sanitizerState">-</strong></div>
          <div class="status-row"><span>Web Search</span><strong id="searchState">-</strong></div>
        </div>
        <div id="chipStack" class="chip-stack"></div>
      </article>

      <article class="card" style="display:grid;gap:8px;">
        <h3 style="margin:0;font-size:16px;">Latest Result</h3>
        <pre id="latestOutput" class="latest-output">No run yet.</pre>
        <button class="btn" id="copyLatestBtn" type="button">Copy Result</button>
      </article>
    </aside>
  </div>
</section>

<script>
  const state = {
    lastResponse: null,
    config: { key_set: false, model: "gpt-4o-mini" },
    running: false,
    stressAbortController: null,
  };

  const timelineEl = document.getElementById("timeline");
  const promptForm = document.getElementById("promptForm");
  const promptInput = document.getElementById("promptInput");
  const runBtn = document.getElementById("runBtn");
  const stressLog = document.getElementById("stressLog");
  const stressSummary = document.getElementById("stressSummary");

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    const div = document.createElement("div");
    div.textContent = String(value);
    return div.innerHTML;
  }

  function nowLabel() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function updateClock() {
    document.getElementById("clockText").textContent = nowLabel();
  }

  function authHeaders(extra = {}) {
    const token = document.getElementById("accessTokenInput").value.trim();
    const headers = { ...extra };
    if (token) headers["Authorization"] = "Bearer " + token;
    return headers;
  }

  function setView(viewId) {
    document.querySelectorAll(".view").forEach((node) => {
      node.classList.toggle("active", node.id === viewId);
    });
    document.querySelectorAll(".nav-item").forEach((node) => {
      node.classList.toggle("active", node.dataset.view === viewId);
    });
  }

  function verdictPill(verdict) {
    const text = verdict || "Unknown";
    const upper = String(text).toUpperCase();
    let cls = "neutral";
    if (upper.includes("PASS")) cls = "ok";
    if (upper.includes("FAIL") || upper.includes("BLOCK")) cls = "bad";
    if (upper.includes("ALLOW") || upper.includes("UNKNOWN")) cls = "warn";
    return '<span class="pill ' + cls + '">' + escapeHtml(text) + '</span>';
  }

  function addTimelineCard(tone, title, bodyHtml, statusText) {
    const card = document.createElement("article");
    card.className = "stage tone-" + tone;
    card.innerHTML =
      '<div class="stage-head"><h4>' + escapeHtml(title) + '</h4>' +
      (statusText ? verdictPill(statusText) : "") +
      "</div>" +
      '<div class="stage-body">' + bodyHtml + "</div>";
    timelineEl.appendChild(card);
    timelineEl.scrollTop = timelineEl.scrollHeight;
    return card;
  }

  function renderClaimTable(claims) {
    if (!claims || claims.length === 0) return '<div class="muted">No claim table returned.</div>';
    let html = '<details><summary>Claim table (' + claims.length + ')</summary>';
    html += '<div class="table-wrap"><table><thead><tr><th>Claim</th><th>Category</th><th>Justification</th></tr></thead><tbody>';
    claims.forEach((item) => {
      html += '<tr><td>' + escapeHtml(item.claim) + '</td><td>' + escapeHtml(item.category) + '</td><td>' + escapeHtml(item.justification) + '</td></tr>';
    });
    html += "</tbody></table></div></details>";
    return html;
  }

  function renderViolations(violations) {
    if (!violations || violations.length === 0) return '<div class="muted">No violations detected.</div>';
    let html = '<ul class="list">';
    violations.forEach((item) => {
      html += "<li>" + escapeHtml(item) + "</li>";
    });
    html += "</ul>";
    return html;
  }

  function renderArbiter(data) {
    let html = '<p><strong>Decision:</strong> ' + escapeHtml(data.arbiter_decision || "-") + "</p>";
    if (data.arbiter_rationale && data.arbiter_rationale.length) {
      html += '<details><summary>Rationale (' + data.arbiter_rationale.length + ')</summary>';
      html += '<ul class="list">' + data.arbiter_rationale.map((x) => "<li>" + escapeHtml(x) + "</li>").join("") + "</ul></details>";
    }
    if (data.arbiter_edits && data.arbiter_edits.length) {
      html += '<details><summary>Edits (' + data.arbiter_edits.length + ')</summary>';
      html += '<ul class="list">';
      data.arbiter_edits.forEach((edit) => {
        const replacement = edit.replacement ? " -> " + escapeHtml(edit.replacement) : "";
        html += "<li><strong>" + escapeHtml(edit.action) + "</strong>: " + escapeHtml(edit.target || "") + replacement + "</li>";
      });
      html += "</ul></details>";
    }
    return html;
  }

  function renderSearchSources(sources) {
    if (!sources || sources.length === 0) return '<div class="muted">No sources found.</div>';
    let html = '<details open><summary>Sources (' + sources.length + ')</summary>';
    html += '<div class="table-wrap"><table><thead><tr><th>#</th><th>Title</th><th>Snippet</th><th>Score</th></tr></thead><tbody>';
    sources.forEach(function(src, idx) {
      html += '<tr><td>[' + (idx + 1) + ']</td>';
      html += '<td><a href="' + escapeHtml(src.url) + '" target="_blank" rel="noopener">' + escapeHtml(src.title) + '</a></td>';
      html += '<td>' + escapeHtml(src.snippet ? src.snippet.slice(0, 200) : '') + '</td>';
      html += '<td>' + (src.score ? src.score.toFixed(2) : '-') + '</td></tr>';
    });
    html += '</tbody></table></div></details>';
    return html;
  }

  function setInspector(response) {
    state.lastResponse = response || null;
    const verdict = response ? (response.final_verdict || "-") : "-";
    const arbiter = response ? (response.arbiter_invoked ? (response.arbiter_decision || "Invoked") : "Not used") : "-";
    const sanitizer = response ? (response.sanitizer_applied ? "Applied" : "No") : "-";
    const searchSt = response ? (response.search_performed ? (response.search_sources.length + " sources") : "Off") : "-";

    document.getElementById("verdictState").textContent = verdict;
    document.getElementById("arbiterState").textContent = arbiter;
    document.getElementById("sanitizerState").textContent = sanitizer;
    document.getElementById("searchState").textContent = searchSt;

    const chips = [];
    if (response) {
      chips.push(response.final_verdict || "Unknown");
      if (response.bypassed) chips.push("Bypass");
      if (response.search_performed) chips.push("Search");
      if (response.arbiter_invoked) chips.push("Arbiter");
      if (response.rewrite_occurred) chips.push("Rewrite");
      if (response.sanitizer_applied) chips.push("Sanitized");
    }

    const chipStack = document.getElementById("chipStack");
    chipStack.innerHTML = chips.map((chip) => '<span class="chip">' + escapeHtml(chip) + '</span>').join("");

    const finalText = response && response.final_result ? response.final_result : "No run yet.";
    document.getElementById("latestOutput").textContent = finalText;
  }

  async function parseErrorResponse(response) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text);
      return parsed.detail || parsed.error || text || response.statusText;
    } catch {
      return text || response.statusText;
    }
  }

  async function loadConfig() {
    const keyState = document.getElementById("keyState");
    const modelState = document.getElementById("modelState");
    const status = document.getElementById("configStatus");
    try {
      const response = await fetch("/api/openai/config", { headers: authHeaders() });
      if (!response.ok) throw new Error(await parseErrorResponse(response));
      const data = await response.json();
      state.config = data;
      if (data.key_set) {
        keyState.textContent = data.key_preview || "Set";
        modelState.textContent = data.model || "-";
        document.getElementById("modelSelect").value = data.model || "gpt-4o-mini";
        status.textContent = "Key configured on server.";
      } else {
        keyState.textContent = "Not set";
        modelState.textContent = data.model || "gpt-4o-mini";
        status.textContent = "No key set. Save a key to run pipeline.";
      }
    } catch (error) {
      keyState.textContent = "Error";
      modelState.textContent = "-";
      status.textContent = "Failed to load config: " + error.message;
    }
  }

  async function saveConfig() {
    const key = document.getElementById("apiKeyInput").value.trim();
    const model = document.getElementById("modelSelect").value;
    const status = document.getElementById("configStatus");
    if (!key) {
      status.textContent = "Enter API key first.";
      return;
    }

    try {
      const response = await fetch("/api/openai/config", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ api_key: key, model })
      });
      if (!response.ok) throw new Error(await parseErrorResponse(response));
      document.getElementById("apiKeyInput").value = "";
      status.textContent = "Saved.";
      await loadConfig();
    } catch (error) {
      status.textContent = "Save failed: " + error.message;
    }
  }

  async function loadTavilyConfig() {
    const status = document.getElementById("tavilyStatus");
    try {
      const response = await fetch("/api/tavily/config", { headers: authHeaders() });
      if (!response.ok) throw new Error(await parseErrorResponse(response));
      const data = await response.json();
      if (data.key_set) {
        document.getElementById("tavilyEnabledToggle").checked = data.enabled;
        status.textContent = "Tavily key configured. " + (data.enabled ? "Search enabled." : "Search disabled.");
      } else {
        document.getElementById("tavilyEnabledToggle").checked = false;
        status.textContent = "No Tavily key set. Enter key to enable web search.";
      }
    } catch (error) {
      status.textContent = "Failed to load Tavily config: " + error.message;
    }
  }

  async function saveTavilyConfig() {
    const key = document.getElementById("tavilyKeyInput").value.trim();
    const enabled = document.getElementById("tavilyEnabledToggle").checked;
    const status = document.getElementById("tavilyStatus");
    if (!key) {
      status.textContent = "Enter Tavily API key first.";
      return;
    }
    try {
      const response = await fetch("/api/tavily/config", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ api_key: key, enabled: enabled })
      });
      if (!response.ok) throw new Error(await parseErrorResponse(response));
      document.getElementById("tavilyKeyInput").value = "";
      status.textContent = "Saved.";
      await loadTavilyConfig();
    } catch (error) {
      status.textContent = "Save failed: " + error.message;
    }
  }

  async function toggleTavilyEnabled() {
    const enabled = document.getElementById("tavilyEnabledToggle").checked;
    const status = document.getElementById("tavilyStatus");
    try {
      const response = await fetch("/api/tavily/toggle?enabled=" + enabled, {
        method: "POST",
        headers: authHeaders(),
      });
      if (!response.ok) throw new Error(await parseErrorResponse(response));
      status.textContent = enabled ? "Search enabled." : "Search disabled.";
    } catch (error) {
      status.textContent = "Toggle failed: " + error.message;
      document.getElementById("tavilyEnabledToggle").checked = !enabled;
    }
  }

  async function runPipeline(event) {
    event.preventDefault();
    if (state.running) return;

    const prompt = promptInput.value.trim();
    if (!prompt) return;

    state.running = true;
    runBtn.disabled = true;
    addTimelineCard("user", "Prompt", '<pre>' + escapeHtml(prompt) + "</pre>", nowLabel());
    promptInput.value = "";

    const loadingCard = addTimelineCard("generator", "Running", '<div class="loading"><span class="loading-dot"></span><span>Calling pipeline API...</span></div>', "In progress");

    try {
      const response = await fetch("/api/pipeline", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          prompt,
          gpt1_system: document.getElementById("g1s").value,
          gpt2_system: document.getElementById("g2s").value,
          gpt3_system: document.getElementById("g3s").value,
        })
      });

      loadingCard.remove();

      if (!response.ok) {
        throw new Error(await parseErrorResponse(response));
      }

      const data = await response.json();

      if (data.search_performed && data.search_sources && data.search_sources.length > 0) {
        addTimelineCard(
          "search",
          "Web Search",
          '<p class="muted">Query: ' + escapeHtml(data.search_query || data.gpt1_input) + '</p>' +
          renderSearchSources(data.search_sources),
          data.search_sources.length + " sources"
        );
      }

      addTimelineCard("generator", "GPT-1 Generator", '<pre>' + escapeHtml(data.gpt1_output || "") + "</pre>", "Done");

      if (data.bypassed) {
        addTimelineCard("final-ok", "Bypass Result", '<p>Activation phrase detected. Verification bypassed.</p><pre>' + escapeHtml(data.final_result || "") + "</pre>", data.final_verdict || "PASS");
        setInspector(data);
        return;
      }

      const verifierBody =
        '<p><strong>Verifier verdict:</strong> ' + escapeHtml(data.gpt2_verdict || "-") + '</p>' +
        renderClaimTable(data.claim_table) +
        renderViolations(data.violations);
      addTimelineCard("verifier", "GPT-2 Verifier", verifierBody, data.gpt2_verdict || "-");

      if (data.arbiter_invoked) {
        addTimelineCard("arbiter", "GPT-3 Arbiter", renderArbiter(data), data.arbiter_decision || "-");
      }

      if (data.rewrite_occurred) {
        addTimelineCard("generator", "Rewrite Output", '<pre>' + escapeHtml(data.rewrite_output || "") + "</pre>", data.rewrite_verdict || "-");
        addTimelineCard("verifier", "Re-Verification", renderClaimTable(data.rewrite_claim_table) + renderViolations(data.rewrite_violations), data.rewrite_verdict || "-");
      }

      const isPass = String(data.final_verdict || "").toUpperCase().includes("PASS");
      addTimelineCard(
        isPass ? "final-ok" : "final-fail",
        "Final Result",
        '<pre>' + escapeHtml(data.final_result || "NO PASS") + "</pre>",
        data.final_verdict || "-"
      );

      if (data.sanitizer_applied && data.gpt1_output_sanitized && data.gpt1_output_sanitized !== data.gpt1_output) {
        addTimelineCard("generator", "Sanitizer Output", '<pre>' + escapeHtml(data.gpt1_output_sanitized) + "</pre>", "Applied");
      }

      setInspector(data);
    } catch (error) {
      loadingCard.remove();
      addTimelineCard("error", "Pipeline Error", '<pre>' + escapeHtml(error.message) + "</pre>", "Failed");
    } finally {
      state.running = false;
      runBtn.disabled = false;
      promptInput.focus();
    }
  }

  function appendStressLine(text, tone = "") {
    const row = document.createElement("div");
    row.className = "log-line " + tone;
    row.textContent = text;
    stressLog.appendChild(row);
    stressLog.scrollTop = stressLog.scrollHeight;
  }

  function renderStressSummary(data) {
    const pss = data.pss || {};
    const metrics = pss.metrics || {};
    stressSummary.innerHTML =
      '<div class="summary-grid">' +
      '<div class="summary-box"><div class="label">PSS Score</div><div class="value">' + Number(pss.score || 0).toFixed(1) + "</div></div>" +
      '<div class="summary-box"><div class="label">Total Tests</div><div class="value">' + (data.total_tests || 0) + "</div></div>" +
      '<div class="summary-box"><div class="label">Pass</div><div class="value">' + (data.total_pass || 0) + "</div></div>" +
      '<div class="summary-box"><div class="label">Fail</div><div class="value">' + (data.total_fail || 0) + "</div></div>" +
      '<div class="summary-box"><div class="label">HLR</div><div class="value">' + ((metrics.HLR || 0) * 100).toFixed(1) + "%</div></div>" +
      '<div class="summary-box"><div class="label">HPR</div><div class="value">' + ((metrics.HPR || 0) * 100).toFixed(1) + "%</div></div>" +
      "</div>";
  }

  async function loadStressCategories() {
    const select = document.getElementById("stressCategory");
    select.innerHTML = '<option value="">All categories</option>';
    try {
      const response = await fetch("/api/stress/categories", { headers: authHeaders() });
      if (!response.ok) return;
      const categories = await response.json();
      categories.forEach((cat) => {
        const option = document.createElement("option");
        option.value = cat;
        option.textContent = cat;
        select.appendChild(option);
      });
    } catch (error) {
      appendStressLine("Failed to load categories: " + error.message, "fail");
    }
  }

  async function runStress() {
    if (state.stressAbortController) return;

    const runStressBtn = document.getElementById("runStressBtn");
    const cancelStressBtn = document.getElementById("cancelStressBtn");
    const category = document.getElementById("stressCategory").value;
    const count = parseInt(document.getElementById("stressCount").value || "0", 10);

    stressLog.innerHTML = "";
    stressSummary.innerHTML = "";
    appendStressLine("Starting stress run...", "warn");

    const body = {};
    if (category) body.category = category;
    if (count > 0) body.count = count;

    const controller = new AbortController();
    state.stressAbortController = controller;
    runStressBtn.disabled = true;
    cancelStressBtn.disabled = false;

    try {
      const response = await fetch("/api/stress", {
        method: "POST",
        headers: authHeaders({
          "Content-Type": "application/json",
          "Accept": "application/x-ndjson"
        }),
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(await parseErrorResponse(response));
      }

      if (!response.body) {
        const fallbackText = await response.text();
        appendStressLine("Non-stream response:", "warn");
        appendStressLine(fallbackText.slice(0, 240), "warn");
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        lines.forEach((line) => {
          const trimmed = line.trim();
          if (!trimmed) return;
          let obj;
          try {
            obj = JSON.parse(trimmed);
          } catch {
            appendStressLine(trimmed, "warn");
            return;
          }

          if (obj.type === "progress") {
            const tone = String(obj.verdict || "").toUpperCase().includes("PASS") ? "pass" :
                         String(obj.verdict || "").toUpperCase().includes("FAIL") ? "fail" : "warn";
            appendStressLine(
              "[" + obj.index + "/" + obj.total + "] " + obj.id + " -> " + obj.verdict +
              (obj.arbiter ? " | arbiter: " + obj.arbiter : "") +
              (obj.rewrite ? " | rewrite" : "") +
              " | " + obj.duration_s + "s",
              tone
            );
            return;
          }

          if (obj.type === "summary") {
            appendStressLine("Completed. Rendering summary.", "pass");
            renderStressSummary(obj);
          }
        });
      }
    } catch (error) {
      if (error.name === "AbortError") {
        appendStressLine("Stress run cancelled.", "warn");
      } else {
        appendStressLine("Stress error: " + error.message, "fail");
      }
    } finally {
      state.stressAbortController = null;
      runStressBtn.disabled = false;
      cancelStressBtn.disabled = true;
    }
  }

  function cancelStress() {
    if (state.stressAbortController) {
      state.stressAbortController.abort();
    }
  }

  function clearWorkspace() {
    timelineEl.innerHTML = "";
    setInspector(null);
  }

  function copyLatest() {
    const text = document.getElementById("latestOutput").textContent;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById("copyLatestBtn");
      const old = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = old; }, 1200);
    }).catch(() => {
      appendStressLine("Clipboard write failed.", "warn");
    });
  }

  document.querySelectorAll(".nav-item").forEach((node) => {
    node.addEventListener("click", () => setView(node.dataset.view));
  });

  document.getElementById("openSettingsBtn").addEventListener("click", () => setView("settingsView"));
  document.getElementById("saveConfigBtn").addEventListener("click", saveConfig);
  document.getElementById("saveTavilyBtn").addEventListener("click", saveTavilyConfig);
  document.getElementById("tavilyEnabledToggle").addEventListener("change", toggleTavilyEnabled);
  document.getElementById("clearBtn").addEventListener("click", clearWorkspace);
  document.getElementById("copyLatestBtn").addEventListener("click", copyLatest);
  document.getElementById("runStressBtn").addEventListener("click", runStress);
  document.getElementById("cancelStressBtn").addEventListener("click", cancelStress);
  promptForm.addEventListener("submit", runPipeline);

  updateClock();
  setInterval(updateClock, 15000);
  loadConfig();
  loadTavilyConfig();
  loadStressCategories();
  promptInput.focus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
