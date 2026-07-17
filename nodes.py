"""
All node functions for the Patient Risk Flagging Agent.

Two of these (extract_patient_data, check_medication_safety) are pure,
already-tested logic. The rest are new. Each "_node" wrapper adapts a
pure function to LangGraph's contract: every node takes the full state
dict and returns the full (updated) state dict.
"""

import datetime


# ---------------------------------------------------------------------
# NODE 1 — Extract patient data (calls an LLM — needs OPENAI_API_KEY)
# ---------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a clinical data extraction assistant.
Extract structured fields from the clinical note below.
Return ONLY valid JSON. No extra text, no markdown.
If a field value is unknown, return null.

Return this exact schema:
{{
  "age": number,
  "gender": "string",
  "primary_diagnosis": "string",
  "secondary_diagnosis": [array of strings],
  "medications": [array of strings],
  "hba1c": number or null,
  "egfr": number or null,
  "days_since_last_visit": number
}}"""


def extract_patient_data(raw_clinical_note: str) -> dict:
    """Pure function: text in, structured dict out. Requires an LLM call."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "{note}"),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"note": raw_clinical_note})


def extract_patient_data_node(state: dict) -> dict:
    """Graph node wrapper: pulls raw_clinical_note from state, merges result back in."""
    extracted = extract_patient_data(state["raw_clinical_note"])
    state.update(extracted)
    return state


# ---------------------------------------------------------------------
# NODE 2 — Retrieve guidelines (simplified stand-in for real RAG)
# ---------------------------------------------------------------------

GUIDELINE_SNIPPETS = {
    "diabetes": "ADA Standards of Care: Target HbA1c below 7.0%. Metformin is "
                "contraindicated when eGFR falls below 30 due to risk of lactic acidosis.",
    "heart failure": "ACC/AHA Guidelines: CHF patients require close monitoring of "
                      "renal function and electrolytes, particularly when on diuretics.",
    "kidney disease": "KDIGO Guidelines: Medication dosing must be adjusted by eGFR. "
                       "Stage 4 CKD requires renal dosing review for all affected drugs.",
    "hypertension": "JNC 8 Guidelines: Target blood pressure below 140/90 for most adults.",
    "asthma": "GINA Guidelines: Reliever inhaler use more than twice weekly indicates "
              "inadequate control and warrants step-up therapy.",
}


def retrieve_guidelines(state: dict) -> dict:
    """
    Real RAG retrieval. Searches the Chroma vector store built by
    build_vectorstore.py for guideline text relevant to this patient's
    diagnosis, using semantic similarity instead of keyword matching.
    Requires build_vectorstore.py to have been run first.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./guideline_chroma_db",
        embedding_function=embeddings,
    )

    query = " ".join(filter(None, [
        state.get("primary_diagnosis", "") or "",
        " ".join(state.get("secondary_diagnosis", []) or []),
    ]))

    results = vectorstore.similarity_search(query, k=2)
    state["guideline_context"] = (
        "\n\n---\n\n".join(doc.page_content for doc in results) if results else "No specific guideline matched."
    )
    return state


def retrieve_guidelines_keyword_fallback(state: dict) -> dict:
    """
    The original keyword-matching version. Kept as a fallback —
    swap this back in for retrieve_guidelines if the vector store
    setup isn't working yet.
    """
    diagnosis_text = " ".join(filter(None, [
        state.get("primary_diagnosis", "") or "",
        " ".join(state.get("secondary_diagnosis", []) or []),
    ])).lower()

    matched = [snippet for keyword, snippet in GUIDELINE_SNIPPETS.items() if keyword in diagnosis_text]
    state["guideline_context"] = " ".join(matched) if matched else "No specific guideline matched."
    return state


# ---------------------------------------------------------------------
# NODE 3 — Medication safety check
# ---------------------------------------------------------------------

def check_medication_safety(patient: dict) -> list:
    """Pure function, already tested against all 6 mock patients."""
    flags = []
    medications = patient.get("medications", [])
    egfr = patient.get("egfr")
    days_since_last_visit = patient.get("days_since_last_visit", 0)
    diagnosis = (patient.get("primary_diagnosis") or "").lower()

    if egfr is not None and egfr < 30:
        for med in medications:
            med_lower = med.lower()
            if "metformin" in med_lower:
                flags.append("Metformin contraindicated — eGFR below 30, risk of lactic acidosis")
            if "digoxin" in med_lower:
                flags.append("Digoxin requires dose adjustment — eGFR below 30 increases risk of digoxin toxicity")

    high_complexity_terms = ["chronic kidney disease", "ckd", "congestive heart failure"]
    is_complex = any(term in diagnosis for term in high_complexity_terms)
    if is_complex and days_since_last_visit > 90:
        flags.append(
            f"Care gap of {days_since_last_visit} days exceeds safe follow-up "
            f"window for {patient.get('primary_diagnosis')} patient"
        )

    return flags


def safety_check_node(state: dict) -> dict:
    """Graph node wrapper around the pure safety-check function."""
    state["safety_flags"] = check_medication_safety(state)
    return state


# ---------------------------------------------------------------------
# NODE 4 — Risk scoring
# ---------------------------------------------------------------------

def score_risk(state: dict) -> dict:
    """A medication safety flag is automatically High. Otherwise score HbA1c + care gap."""
    safety_flags = state.get("safety_flags", [])
    hba1c = state.get("hba1c")
    days_since_last_visit = state.get("days_since_last_visit", 0)

    if safety_flags:
        state["risk_level"] = "High"
        state["risk_reason"] = "; ".join(safety_flags)
        return state

    score = 0
    reasons = []

    if hba1c is not None:
        if hba1c >= 9.0:
            score += 2
            reasons.append(f"HbA1c of {hba1c} indicates poorly controlled diabetes")
        elif hba1c >= 8.0:
            score += 1
            reasons.append(f"HbA1c of {hba1c} is above target")

    if days_since_last_visit > 90:
        score += 2
        reasons.append(f"{days_since_last_visit} days since last visit is significantly overdue")
    elif days_since_last_visit > 60:
        score += 1
        reasons.append(f"{days_since_last_visit} days since last visit warrants a check-in")

    if score >= 3:
        state["risk_level"] = "High"
    elif score >= 1:
        state["risk_level"] = "Medium"
    else:
        state["risk_level"] = "Low"

    state["risk_reason"] = "; ".join(reasons) if reasons else "No significant risk factors identified"
    return state


def route_after_scoring(state: dict) -> str:
    """Conditional edge: decide whether this case needs a human checkpoint."""
    if state.get("safety_flags") or state.get("risk_level") == "High":
        return "human_review"
    return "generate_output"


# ---------------------------------------------------------------------
# NODE 5 — Human review checkpoint
# ---------------------------------------------------------------------

def human_review_checkpoint(state: dict) -> dict:
    """
    Simplified stand-in for a real checkpoint. A production version would use
    LangGraph's interrupt() with a persistent checkpointer to actually pause
    execution and wait for a person. This version logs the case clearly and
    marks it reviewed, so your pipeline runs end to end while you build the
    real interrupt logic later.
    """
    print("\n--- HUMAN REVIEW CHECKPOINT ---")
    print(f"Patient: {state.get('patient_id')}")
    print(f"Risk Level: {state.get('risk_level')}")
    print(f"Reason: {state.get('risk_reason')}")
    print("--- Reviewed and approved (simulated) ---\n")

    state["needs_human_review"] = True
    return state


# ---------------------------------------------------------------------
# NODE 6 — Generate structured output
# ---------------------------------------------------------------------

def generate_output(state: dict) -> dict:
    risk_level = state.get("risk_level")
    if risk_level == "High":
        action = "Urgent follow-up required"
    elif risk_level == "Medium":
        action = "Schedule routine follow-up"
    else:
        action = "Continue routine care"

    state["final_output"] = {
        "patient_id": state.get("patient_id"),
        "risk_level": risk_level,
        "risk_reason": state.get("risk_reason"),
        "guideline_context": state.get("guideline_context"),
        "safety_flags": state.get("safety_flags", []),
        "reviewed_by_human": state.get("needs_human_review", False),
        "recommended_action": action,
    }
    return state


# ---------------------------------------------------------------------
# NODE 7 — Log to audit trail
# ---------------------------------------------------------------------

AUDIT_LOG = []


def log_audit(state: dict) -> dict:
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "patient_id": state.get("patient_id"),
        "risk_level": state.get("risk_level"),
        "safety_flags": state.get("safety_flags", []),
        "reviewed_by_human": state.get("needs_human_review", False),
    }
    AUDIT_LOG.append(entry)
    print(f"Logged audit entry for {state.get('patient_id')}")
    return state
