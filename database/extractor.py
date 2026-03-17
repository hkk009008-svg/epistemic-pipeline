import json
import logging
from pipeline.helpers import call_llm
from database.client import insert_claim


def extract_and_store_claims(claim_table, llm_config):
    """
    Given a list of verified ClaimEntry objects, extracts Subject-Relation-Object
    triples for 'Observed' claims and stores them in the Knowledge Graph.
    """
    observed_claims = []
    for ct in claim_table:
        cat = ct.category if isinstance(ct.category, str) else ""
        if cat.lower().strip() == "observed":
            observed_claims.append(ct.claim)

    if not observed_claims:
        return

    system_prompt = (
        "You are an epistemic extraction engine. Your job is to convert natural language claims into strict Subject-Relation-Object triples.\n"
        "Return ONLY a JSON array of objects with 'subject', 'relation', 'object', and 'original_text' keys.\n"
        "Keep the subject and object concise (usually 1-3 words).\n"
        "Example:\n"
        '[\n  {"subject": "staking income", "relation": "is considered", "object": "taxable", "original_text": "staking income is considered taxable in the United States"}\n]'
    )

    user_prompt = (
        "Convert the following verified facts into structural triples:\n"
        + "\n".join(f"- {claim}" for claim in observed_claims)
    )

    try:
        raw_json = call_llm(llm_config, system_prompt, user_prompt, expect_json=True)
        triples = json.loads(raw_json)

        for t in triples:
            subj = t.get("subject")
            rel = t.get("relation")
            obj = t.get("object")
            if subj and rel and obj:
                insert_claim(
                    subject_name=subj,
                    relation=rel,
                    object_name=obj,
                    confidence="High",
                    original_text=t.get("original_text", ""),
                )
    except Exception as e:
        logging.error(f"Failed to extract and store claims to Knowledge Graph: {e}")
