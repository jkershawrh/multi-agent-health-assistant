"""A2A-compliant agent template -- serves agent card and handles JSON-RPC tasks.

Each agent instance is configured via environment variables:
  AGENT_NAME  -- agent identity (triage, clinical, scheduling)
  AGENT_SKILLS -- comma-separated skill ids
  AGENT_PORT  -- port to listen on (default 8001)

Demo mode: all agents return simulated responses without LLM backends.
"""

import logging
import os
import random
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import models
from auth import TokenAuthMiddleware

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("a2a-agent")

AGENT_NAME = os.environ.get("AGENT_NAME", "generic")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "8001"))
AGENT_SKILLS_RAW = os.environ.get("AGENT_SKILLS", "respond")

MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:1.5b")
DEMO_MODE = os.environ.get("DEMO_MODE", "").lower() in ("true", "1", "yes")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "")

AI_DISCLAIMER = (
    "Agent responses are AI-generated -- verify clinical "
    "recommendations with qualified healthcare professionals."
)

# ---------------------------------------------------------------------------
# Skill and card definitions per agent type
# ---------------------------------------------------------------------------

AGENT_CONFIGS = {
    "triage": {
        "description": (
            "Patient triage agent -- classifies urgency and prioritizes "
            "cases for clinical review. Runs on Intel Xeon CPU."
        ),
        "skills": [
            models.AgentSkill(
                id="classify",
                name="Classify Urgency",
                description="Classify patient case by urgency level (critical, urgent, routine)",
                tags=["triage", "classification"],
                examples=["Classify this patient case", "What is the urgency level?"],
            ),
            models.AgentSkill(
                id="prioritize",
                name="Prioritize Cases",
                description="Rank multiple cases by clinical priority",
                tags=["triage", "prioritization"],
                examples=["Prioritize these cases", "Which patient needs attention first?"],
            ),
        ],
    },
    "clinical": {
        "description": (
            "Clinical analysis agent -- provides diagnostic suggestions "
            "and treatment recommendations. Runs on Intel Xeon CPU."
        ),
        "skills": [
            models.AgentSkill(
                id="diagnose",
                name="Suggest Diagnosis",
                description="Analyze symptoms and suggest possible diagnoses",
                tags=["clinical", "diagnosis"],
                examples=["What could cause these symptoms?", "Suggest a diagnosis"],
            ),
            models.AgentSkill(
                id="recommend",
                name="Recommend Treatment",
                description="Recommend evidence-based treatment options",
                tags=["clinical", "treatment"],
                examples=["What treatment do you recommend?", "Treatment options for this condition"],
            ),
        ],
    },
    "scheduling": {
        "description": (
            "Scheduling agent -- manages appointments and sends follow-up "
            "notifications. Runs on Intel Xeon CPU."
        ),
        "skills": [
            models.AgentSkill(
                id="schedule",
                name="Schedule Appointment",
                description="Find and book the next available appointment slot",
                tags=["scheduling", "appointment"],
                examples=["Schedule a follow-up", "Find the next available slot"],
            ),
            models.AgentSkill(
                id="notify",
                name="Send Notification",
                description="Send appointment reminders and care instructions",
                tags=["scheduling", "notification"],
                examples=["Send a reminder", "Notify the patient about their appointment"],
            ),
        ],
    },
}

# ---------------------------------------------------------------------------
# Demo responses per agent type
# ---------------------------------------------------------------------------

DEMO_RESPONSES = {
    "triage": {
        "classify": (
            "Urgency classification: URGENT. Patient presents with acute symptoms "
            "requiring clinical review within 2 hours. Vital signs indicate elevated "
            "heart rate and blood pressure. Recommend immediate clinical assessment."
        ),
        "prioritize": (
            "Priority ranking complete. Case assigned priority level 2 of 5. "
            "Flagged for expedited clinical review based on symptom severity "
            "and patient history indicators."
        ),
    },
    "clinical": {
        "diagnose": (
            "Differential diagnosis based on presented symptoms: "
            "(1) Acute coronary syndrome -- recommended: ECG, troponin levels. "
            "(2) Gastroesophageal reflux -- recommended: clinical history review. "
            "(3) Musculoskeletal strain -- recommended: physical examination. "
            "Further workup recommended to narrow differential."
        ),
        "recommend": (
            "Treatment recommendation: Begin with non-invasive assessment protocol. "
            "Order baseline labs (CBC, BMP, troponin). Continuous cardiac monitoring. "
            "If cardiac etiology confirmed, initiate guideline-directed medical therapy. "
            "Schedule follow-up in 48 hours."
        ),
    },
    "scheduling": {
        "schedule": (
            "Appointment scheduled: Follow-up visit booked for next available slot "
            "(within 48 hours). Provider: Dr. Martinez, Internal Medicine. "
            "Location: Clinic B, Room 204. Duration: 30 minutes."
        ),
        "notify": (
            "Notification sent: Patient notified via preferred contact method. "
            "Appointment confirmation and pre-visit instructions delivered. "
            "Reminder scheduled for 24 hours before appointment."
        ),
    },
}


def _get_agent_config() -> dict:
    """Return the agent configuration for the current AGENT_NAME."""
    return AGENT_CONFIGS.get(AGENT_NAME, {
        "description": f"{AGENT_NAME} agent -- generic A2A-compliant agent. Runs on Intel Xeon CPU.",
        "skills": [
            models.AgentSkill(
                id=s.strip(),
                name=s.strip().title(),
                description=f"Perform {s.strip()} operation",
                tags=[AGENT_NAME, s.strip()],
            )
            for s in AGENT_SKILLS_RAW.split(",") if s.strip()
        ],
    })


def _build_agent_card() -> models.AgentCard:
    """Build the agent card from environment configuration."""
    config = _get_agent_config()
    return models.AgentCard(
        name=AGENT_NAME,
        description=config["description"],
        url=f"http://localhost:{AGENT_PORT}",
        skills=config["skills"],
    )


def _demo_response(text: str) -> str:
    """Generate a simulated response based on agent type and query content."""
    agent_responses = DEMO_RESPONSES.get(AGENT_NAME, {})

    # Try to match a skill based on query keywords
    text_lower = text.lower()
    for skill_id, response in agent_responses.items():
        if skill_id in text_lower:
            return response

    # Return the first available response for this agent type
    if agent_responses:
        return next(iter(agent_responses.values()))

    return (
        f"[{AGENT_NAME}] Processed query: {text[:100]}. "
        f"Analysis complete. {AI_DISCLAIMER}"
    )


async def _llm_response(text: str, model_name: str = "") -> str:
    """Call the LLM via the OpenAI-compatible endpoint and return its reply."""
    use_model = model_name or MODEL_NAME
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{MODEL_ENDPOINT}/chat/completions",
                json={
                    "model": use_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are a {AGENT_NAME} agent in a healthcare workflow. "
                                "Provide concise, professional responses. "
                                f"{AI_DISCLAIMER}"
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LLM call failed (%s), falling back to demo response", exc)
        return _demo_response(text)


# ---------------------------------------------------------------------------
# MCP tool calling
# ---------------------------------------------------------------------------

TOOL_KEYWORDS = {
    "lookup_patient_record": ["patient record", "patient history", "medical record", "pat-"],
    "check_drug_interactions": ["drug interaction", "medication interaction", "drug-drug", "medications"],
    "find_available_slots": ["appointment", "schedule", "available slot", "book", "follow-up"],
}


async def _call_mcp_tools(text: str) -> str:
    """Call relevant MCP tools based on query content and return results."""
    if not MCP_SERVER_URL:
        return ""

    text_lower = text.lower()
    tools_to_call = []
    for tool_name, keywords in TOOL_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tools_to_call.append(tool_name)

    if not tools_to_call:
        return ""

    results = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for tool_name in tools_to_call:
                arguments = _build_tool_arguments(tool_name, text)
                resp = await client.post(
                    f"{MCP_SERVER_URL}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": str(uuid.uuid4()),
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("result", {})
                    content = result.get("content", [])
                    for item in content:
                        if item.get("text"):
                            results.append(f"[{tool_name}]: {item['text']}")
                            logger.info("MCP tool call: %s", tool_name)
    except Exception as e:
        logger.warning("MCP tool call failed: %s", e)

    return "\n".join(results)


def _build_tool_arguments(tool_name: str, text: str) -> dict:
    """Extract tool arguments from query text (best-effort for demo)."""
    text_lower = text.lower()
    if tool_name == "lookup_patient_record":
        for token in text.split():
            if token.upper().startswith("PAT-"):
                return {"patient_id": token.upper()}
        return {"patient_id": "PAT-001"}
    elif tool_name == "check_drug_interactions":
        known_drugs = ["metformin", "lisinopril", "aspirin", "warfarin", "albuterol", "potassium"]
        found = [d for d in known_drugs if d in text_lower]
        return {"medications": found if found else ["metformin", "lisinopril"]}
    elif tool_name == "find_available_slots":
        dept = "general"
        for d in ["cardiology", "orthopedics", "general"]:
            if d in text_lower:
                dept = d
                break
        urgency = "routine"
        for u in ["critical", "urgent"]:
            if u in text_lower:
                urgency = u
                break
        return {"department": dept, "urgency": urgency}
    return {}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=f"A2A Agent: {AGENT_NAME}",
    description=f"A2A-compliant {AGENT_NAME} agent for healthcare workflows.",
    version="1.0.0",
)
app.add_middleware(TokenAuthMiddleware)


@app.get("/health")
async def health():
    if MODEL_ENDPOINT and not DEMO_MODE:
        mode = "llm"
    else:
        mode = "demo"
    return {
        "status": "healthy",
        "agent": AGENT_NAME,
        "mode": mode,
    }


@app.get("/.well-known/agent-card.json")
async def agent_card():
    card = _build_agent_card()
    return card.model_dump()


@app.post("/a2a")
async def a2a_endpoint(request: models.JsonRpcRequest):
    """Handle A2A JSON-RPC 2.0 requests."""

    if request.method == "tasks/send":
        params = request.params or {}
        task_id = params.get("id", str(uuid.uuid4()))
        model_override = params.get("model_override", "")
        message = params.get("message", {})
        parts = message.get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        start = time.monotonic()

        tool_context = await _call_mcp_tools(text)
        enriched_text = f"{text}\n\nTool results:\n{tool_context}" if tool_context else text

        if MODEL_ENDPOINT and not DEMO_MODE:
            response_text = await _llm_response(enriched_text, model_override)
        else:
            response_text = _demo_response(text)

        if tool_context:
            response_text = f"{response_text}\n\n[MCP tool data retrieved]\n{tool_context}"

        latency_ms = round((time.monotonic() - start) * 1000, 2)

        # Simulate realistic processing time (demo mode only)
        if not MODEL_ENDPOINT or DEMO_MODE:
            latency_ms = max(latency_ms, round(random.uniform(15, 150), 2))

        logger.info(
            "A2A tasks/send [%s] task=%s latency=%.1fms",
            AGENT_NAME, task_id, latency_ms,
        )

        return models.JsonRpcResponse(
            id=request.id,
            result=models.Task(
                id=task_id,
                contextId=str(uuid.uuid4()),
                status=models.TaskStatus(state="completed"),
                artifacts=[
                    models.Artifact(
                        parts=[models.Part(text=response_text)]
                    )
                ],
            ),
        )

    if request.method == "tasks/get":
        task_id = (request.params or {}).get("id", "unknown")
        return models.JsonRpcResponse(
            id=request.id,
            result=models.Task(
                id=task_id,
                status=models.TaskStatus(state="completed"),
            ),
        )

    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {request.method}",
            },
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
