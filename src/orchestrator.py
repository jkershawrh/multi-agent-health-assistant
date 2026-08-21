"""Multi-Agent Orchestrator -- discovers and delegates to A2A agents.

Discovers agents via HTTP GET to /.well-known/agent-card.json,
maintains a registry, and executes multi-step workflows by
delegating tasks to agents via the A2A JSON-RPC protocol.

Demo mode: 3 built-in simulated agents (triage, clinical, scheduling)
respond without real LLM backends.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI

import models

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("orchestrator")

AI_DISCLAIMER = (
    "Agent responses are AI-generated -- verify clinical "
    "recommendations with qualified healthcare professionals."
)

# Agent URLs to discover on startup (comma-separated)
AGENT_URLS = os.environ.get(
    "AGENT_URLS",
    "http://triage-agent:8001,http://clinical-agent:8002,http://scheduling-agent:8003",
)


# ---------------------------------------------------------------------------
# A2A Client
# ---------------------------------------------------------------------------


class A2AClient:
    """Discovers and communicates with A2A-compliant agents."""

    def __init__(self):
        self.agents: Dict[str, models.DiscoveredAgent] = {}

    async def discover(self, base_url: str) -> Optional[models.DiscoveredAgent]:
        """Fetch /.well-known/agent-card.json and register the agent."""
        url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                card_data = resp.json()

            skills = [
                models.AgentSkill(**s)
                for s in card_data.get("skills", [])
            ]
            agent = models.DiscoveredAgent(
                name=card_data.get("name", "unknown"),
                url=base_url.rstrip("/"),
                status="active",
                skills=skills,
            )
            self.agents[agent.name] = agent
            logger.info("Discovered agent: %s at %s (%d skills)",
                        agent.name, base_url, len(skills))
            return agent
        except Exception as e:
            logger.warning("Discovery failed for %s: %s", base_url, e)
            return None

    async def send_task(self, agent_name: str, text: str) -> dict:
        """Send a tasks/send JSON-RPC request to a discovered agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return {"error": f"Agent not found: {agent_name}"}

        task_id = str(uuid.uuid4())
        rpc_request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                },
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{agent.url}/a2a",
                    json=rpc_request,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("Task send to %s failed: %s", agent_name, e)
            return {"error": str(e)}

    def list_agents(self) -> List[models.DiscoveredAgent]:
        return list(self.agents.values())

    def get_agent(self, name: str) -> Optional[models.DiscoveredAgent]:
        return self.agents.get(name)


# ---------------------------------------------------------------------------
# Workflow engine
# ---------------------------------------------------------------------------

# Workflow definitions: each maps to an ordered list of (agent, action) pairs
WORKFLOW_DEFINITIONS = {
    "patient_triage": [
        ("triage", "classify"),
        ("clinical", "diagnose"),
        ("scheduling", "schedule"),
    ],
    "general": [
        ("triage", "classify"),
        ("clinical", "recommend"),
        ("scheduling", "notify"),
    ],
}


async def execute_workflow(
    a2a_client: A2AClient,
    query: str,
    workflow_type: str = "general",
) -> models.WorkflowResponse:
    """Execute a multi-agent workflow by delegating tasks sequentially."""
    steps_config = WORKFLOW_DEFINITIONS.get(
        workflow_type,
        WORKFLOW_DEFINITIONS["general"],
    )

    steps: List[models.WorkflowStep] = []
    agents_involved: List[str] = []
    total_start = time.monotonic()

    # Build context that accumulates across steps
    context = query
    for agent_name, action in steps_config:
        step_start = time.monotonic()

        # Include previous step results in the context
        task_text = f"[{action}] {context}"
        result = await a2a_client.send_task(agent_name, task_text)

        step_latency = round((time.monotonic() - step_start) * 1000, 2)

        # Extract result text from A2A response
        result_text = _extract_result_text(result)

        steps.append(models.WorkflowStep(
            agent=agent_name,
            action=action,
            result=result_text,
            latency_ms=step_latency,
        ))
        agents_involved.append(agent_name)

        # Accumulate context for next step
        context = f"{context}\n\nPrevious step ({agent_name}/{action}): {result_text}"

    total_latency = round((time.monotonic() - total_start) * 1000, 2)

    return models.WorkflowResponse(
        steps=steps,
        total_latency_ms=total_latency,
        agents_involved=list(dict.fromkeys(agents_involved)),
        ai_disclaimer=AI_DISCLAIMER,
    )


def _extract_result_text(rpc_response: dict) -> str:
    """Extract text from A2A JSON-RPC response."""
    if "error" in rpc_response:
        return f"Error: {rpc_response['error']}"

    result = rpc_response.get("result", {})
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return "No result returned"

    texts = []
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("text"):
                texts.append(part["text"])

    return " ".join(texts) if texts else "No text in response"


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

a2a_client = A2AClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Discover agents on startup."""
    urls = [u.strip() for u in AGENT_URLS.split(",") if u.strip()]
    logger.info("Discovering %d agents...", len(urls))

    # Try discovery with retries for startup ordering
    for attempt in range(3):
        for url in urls:
            if not any(a.url == url.rstrip("/") for a in a2a_client.list_agents()):
                await a2a_client.discover(url)

        discovered = len(a2a_client.list_agents())
        if discovered >= len(urls):
            break

        if attempt < 2:
            wait = 2 * (attempt + 1)
            logger.info(
                "Discovered %d/%d agents, retrying in %ds...",
                discovered, len(urls), wait,
            )
            await asyncio.sleep(wait)

    logger.info(
        "Agent discovery complete: %d agents registered",
        len(a2a_client.list_agents()),
    )
    yield


app = FastAPI(
    title="Multi-Agent Health Assistant Orchestrator",
    description=(
        "Orchestrates cooperating AI agents using the A2A protocol "
        "for healthcare workflows. Runs on Intel Xeon CPU."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    agents = a2a_client.list_agents()
    return {
        "status": "healthy",
        "agents_discovered": len(agents),
        "agent_names": [a.name for a in agents],
    }


@app.get("/.well-known/agent-card.json")
async def orchestrator_agent_card():
    """Serve the orchestrator's own A2A agent card."""
    card = models.AgentCard(
        name="orchestrator",
        description=(
            "Multi-agent orchestrator -- discovers and coordinates healthcare "
            "AI agents using A2A protocol. Runs on Intel Xeon CPU."
        ),
        url="http://localhost:8000",
        skills=[
            models.AgentSkill(
                id="orchestrate-workflow",
                name="Orchestrate Workflow",
                description="Execute a multi-agent healthcare workflow",
                tags=["orchestration", "workflow", "multi-agent"],
                examples=[
                    "Run a patient triage workflow",
                    "Coordinate care for this patient",
                ],
            ),
        ],
    )
    return card.model_dump()


@app.get("/api/v1/agents")
async def list_agents():
    """List all discovered agents."""
    agents = a2a_client.list_agents()
    return {
        "agents": [a.model_dump() for a in agents],
        "count": len(agents),
    }


@app.post("/api/v1/workflow")
async def run_workflow(request: models.WorkflowRequest):
    """Execute a multi-agent workflow."""
    return await execute_workflow(
        a2a_client,
        query=request.query,
        workflow_type=request.workflow_type,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
