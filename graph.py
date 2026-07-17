"""
LangGraph wiring for the Patient Risk Flagging Agent.

Run this in your own notebook/environment where langgraph and
langchain are installed. This file assumes nodes.py is in the
same folder.
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from nodes import (
    extract_patient_data_node,
    retrieve_guidelines,
    safety_check_node,
    score_risk,
    human_review_checkpoint,
    generate_output,
    log_audit,
    route_after_scoring,
)


class PatientRiskState(TypedDict):
    patient_id: str
    raw_clinical_note: str
    age: int
    gender: str
    primary_diagnosis: str
    secondary_diagnosis: List[str]
    medications: List[str]
    hba1c: Optional[float]
    egfr: Optional[float]
    days_since_last_visit: int
    guideline_context: str
    safety_flags: List[str]
    risk_level: str
    risk_reason: str
    needs_human_review: bool
    final_output: dict


graph = StateGraph(PatientRiskState)

graph.add_node("extract", extract_patient_data_node)
graph.add_node("retrieve_guidelines", retrieve_guidelines)
graph.add_node("safety_check", safety_check_node)
graph.add_node("risk_scoring", score_risk)
graph.add_node("human_review", human_review_checkpoint)
graph.add_node("generate_output", generate_output)
graph.add_node("log_audit", log_audit)

graph.set_entry_point("extract")
graph.add_edge("extract", "retrieve_guidelines")
graph.add_edge("retrieve_guidelines", "safety_check")
graph.add_edge("safety_check", "risk_scoring")

graph.add_conditional_edges(
    "risk_scoring",
    route_after_scoring,
    {"human_review": "human_review", "generate_output": "generate_output"},
)

graph.add_edge("human_review", "generate_output")
graph.add_edge("generate_output", "log_audit")
graph.add_edge("log_audit", END)

app = graph.compile()


if __name__ == "__main__":
    import json

    with open("mock_patients.json") as f:
        data = json.load(f)

    # Note: this calls extract_patient_data_node, which calls a real LLM.
    # Set OPENAI_API_KEY in your environment before running this.
    for entry in data["patients"]:
        initial_state = {"patient_id": entry["patient_id"], "raw_clinical_note": entry["raw_clinical_note"]}
        result = app.invoke(initial_state)
        print(json.dumps(result["final_output"], indent=2))
