"""MCP Tool Server -- healthcare tools for agent workflows.

Implements a subset of the Model Context Protocol (MCP) over HTTP,
exposing tools that agents can call for external data during task
processing. All data is simulated for demo purposes.
"""

import logging
import os
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("mcp-server")

MCP_PORT = int(os.environ.get("MCP_PORT", "8004"))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "lookup_patient_record",
        "description": "Look up a patient's medical record by ID. Returns demographics, medical history, allergies, and current medications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier (e.g. PAT-001)",
                },
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "check_drug_interactions",
        "description": "Check for known drug-drug interactions between a list of medications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medication names to check for interactions",
                },
            },
            "required": ["medications"],
        },
    },
    {
        "name": "find_available_slots",
        "description": "Find available appointment slots by department and urgency level.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "Department name (e.g. cardiology, general, orthopedics)",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["routine", "urgent", "critical"],
                    "description": "Urgency level for appointment scheduling",
                },
            },
            "required": ["department"],
        },
    },
]

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

PATIENT_RECORDS = {
    "PAT-001": {
        "name": "Jane Doe",
        "age": 62,
        "sex": "F",
        "blood_type": "A+",
        "conditions": ["hypertension", "type 2 diabetes"],
        "allergies": ["penicillin", "sulfa drugs"],
        "medications": ["metformin 500mg", "lisinopril 10mg", "aspirin 81mg"],
        "last_visit": "2026-07-15",
    },
    "PAT-002": {
        "name": "John Smith",
        "age": 45,
        "sex": "M",
        "blood_type": "O-",
        "conditions": ["asthma"],
        "allergies": [],
        "medications": ["albuterol inhaler"],
        "last_visit": "2026-08-01",
    },
}

DRUG_INTERACTIONS = {
    ("metformin", "lisinopril"): {
        "severity": "low",
        "description": "May enhance hypoglycemic effect. Monitor blood glucose.",
    },
    ("warfarin", "aspirin"): {
        "severity": "high",
        "description": "Increased risk of bleeding. Requires close INR monitoring.",
    },
    ("lisinopril", "potassium"): {
        "severity": "moderate",
        "description": "Risk of hyperkalemia. Monitor potassium levels.",
    },
}

APPOINTMENT_SLOTS = {
    "cardiology": {
        "routine": [
            {"date": "2026-08-25", "time": "09:00", "provider": "Dr. Chen"},
            {"date": "2026-08-26", "time": "14:30", "provider": "Dr. Chen"},
        ],
        "urgent": [
            {"date": "2026-08-22", "time": "11:00", "provider": "Dr. Patel"},
        ],
        "critical": [
            {"date": "2026-08-21", "time": "ASAP", "provider": "Dr. Patel (on-call)"},
        ],
    },
    "general": {
        "routine": [
            {"date": "2026-08-23", "time": "10:00", "provider": "Dr. Martinez"},
            {"date": "2026-08-24", "time": "15:00", "provider": "Dr. Lee"},
        ],
        "urgent": [
            {"date": "2026-08-22", "time": "08:30", "provider": "Dr. Martinez"},
        ],
        "critical": [
            {"date": "2026-08-21", "time": "ASAP", "provider": "Dr. Lee (on-call)"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _lookup_patient_record(arguments: dict) -> dict:
    patient_id = arguments.get("patient_id", "")
    record = PATIENT_RECORDS.get(patient_id)
    if not record:
        return {"error": f"Patient {patient_id} not found", "available_ids": list(PATIENT_RECORDS.keys())}
    return {"patient_id": patient_id, **record}


def _check_drug_interactions(arguments: dict) -> dict:
    medications = arguments.get("medications", [])
    meds_lower = [m.split()[0].lower() for m in medications]
    found = []
    for (drug_a, drug_b), interaction in DRUG_INTERACTIONS.items():
        if drug_a in meds_lower and drug_b in meds_lower:
            found.append({"drugs": [drug_a, drug_b], **interaction})
        elif drug_b in meds_lower and drug_a in meds_lower:
            found.append({"drugs": [drug_a, drug_b], **interaction})
    return {
        "medications_checked": medications,
        "interactions_found": len(found),
        "interactions": found,
    }


def _find_available_slots(arguments: dict) -> dict:
    department = arguments.get("department", "general").lower()
    urgency = arguments.get("urgency", "routine").lower()
    dept_slots = APPOINTMENT_SLOTS.get(department, APPOINTMENT_SLOTS.get("general", {}))
    slots = dept_slots.get(urgency, dept_slots.get("routine", []))
    return {
        "department": department,
        "urgency": urgency,
        "slots": slots,
        "count": len(slots),
    }


TOOL_HANDLERS = {
    "lookup_patient_record": _lookup_patient_record,
    "check_drug_interactions": _check_drug_interactions,
    "find_available_slots": _find_available_slots,
}


# ---------------------------------------------------------------------------
# MCP JSON-RPC endpoint
# ---------------------------------------------------------------------------


class McpRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[dict] = None


app = FastAPI(
    title="MCP Tool Server",
    description="Healthcare tools for multi-agent workflows (MCP protocol).",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tools_available": len(TOOLS),
        "tool_names": [t["name"] for t in TOOLS],
    }


@app.post("/mcp")
async def mcp_endpoint(request: McpRequest):
    """Handle MCP JSON-RPC requests (tools/list and tools/call)."""
    if request.method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "result": {"tools": TOOLS},
        }

    if request.method == "tools/call":
        params = request.params or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request.id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            })

        logger.info("MCP tools/call [%s] args=%s", tool_name, arguments)
        result = handler(arguments)

        return {
            "jsonrpc": "2.0",
            "id": request.id,
            "result": {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            },
        }

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": request.id,
        "error": {"code": -32601, "message": f"Method not found: {request.method}"},
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT)
