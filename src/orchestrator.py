"""Multi-Agent Orchestrator -- discovers and delegates to A2A agents.

Discovers agents via HTTP GET to /.well-known/agent-card.json,
maintains a registry, and executes multi-step workflows by
delegating tasks to agents via the A2A JSON-RPC protocol.

Demo mode: 3 built-in simulated roles (triage, clinical, scheduling)
demonstrate message routing without medical output or real integrations.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

import models

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("orchestrator")

AI_DISCLAIMER = (
    "Educational simulation only. It does not provide medical advice, diagnosis, "
    "treatment, triage, scheduling, or emergency services. Use synthetic data only."
)

# Agent URLs to discover on startup (comma-separated)
AGENT_URLS = os.environ.get(
    "AGENT_URLS",
    "http://triage-agent:8001,http://clinical-agent:8002,http://scheduling-agent:8003",
)
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "8000"))

if MAX_CONTEXT_CHARS < 2_000:
    raise ValueError("MAX_CONTEXT_CHARS must be at least 2000")


# ---------------------------------------------------------------------------
# A2A Client
# ---------------------------------------------------------------------------


class A2AClient:
    """Discovers and communicates with agents using the teaching subset."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.agents: dict[str, models.DiscoveredAgent] = {}
        self.transport = transport

    async def discover(self, base_url: str) -> models.DiscoveredAgent | None:
        """Fetch /.well-known/agent-card.json and register the agent."""
        url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
        try:
            async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                card = models.AgentCard.model_validate(resp.json())

            agent = models.DiscoveredAgent(
                name=card.name,
                url=base_url.rstrip("/"),
                status="active",
                skills=card.skills,
            )
            self.agents[agent.name] = agent
            logger.info("Discovered agent: %s at %s (%d skills)",
                        agent.name, base_url, len(agent.skills))
            return agent
        except (httpx.HTTPError, ValidationError, ValueError) as e:
            logger.warning("Discovery failed for %s: %s", base_url, e)
            return None

    async def send_task(self, agent_name: str, text: str) -> dict:
        """Send a message/send JSON-RPC request to a discovered agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return {"error": f"Agent not found: {agent_name}"}

        task_id = str(uuid.uuid4())
        rpc_request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
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
            async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
                resp = await client.post(
                    f"{agent.url}/a2a",
                    json=rpc_request,
                )
                resp.raise_for_status()
                rpc_response = models.JsonRpcResponse.model_validate(resp.json())
                payload = rpc_response.model_dump(exclude_none=True)
                if rpc_response.error:
                    return payload
                task = rpc_response.result
                if (
                    task is None
                    or task.status.state != "completed"
                    or not task.artifacts
                ):
                    return {"error": "Agent returned no completed artifact"}
                return payload
        except (ValidationError, ValueError) as e:
            logger.error("Invalid response from %s: %s", agent_name, e)
            return {"error": "Agent returned an invalid response"}
        except httpx.HTTPError as e:
            logger.error("Task send to %s failed: %s", agent_name, e)
            return {"error": "Agent request failed"}

    def list_agents(self) -> list[models.DiscoveredAgent]:
        return list(self.agents.values())

    def get_agent(self, name: str) -> models.DiscoveredAgent | None:
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
    if workflow_type not in WORKFLOW_DEFINITIONS:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    steps_config = WORKFLOW_DEFINITIONS[workflow_type]

    steps: list[models.WorkflowStep] = []
    agents_involved: list[str] = []
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

        if result.get("error"):
            return models.WorkflowResponse(
                steps=steps,
                total_latency_ms=round((time.monotonic() - total_start) * 1000, 2),
                agents_involved=list(dict.fromkeys(agents_involved)),
                status="failed",
                failed_step=f"{agent_name}/{action}",
                ai_disclaimer=AI_DISCLAIMER,
            )

        # Accumulate context for next step
        context = (
            f"{context}\n\nPrevious step ({agent_name}/{action}): {result_text}"
        )[-MAX_CONTEXT_CHARS:]

    total_latency = round((time.monotonic() - total_start) * 1000, 2)

    return models.WorkflowResponse(
        steps=steps,
        total_latency_ms=total_latency,
        agents_involved=list(dict.fromkeys(agents_involved)),
        status="completed",
        ai_disclaimer=AI_DISCLAIMER,
    )


def _extract_result_text(rpc_response: dict) -> str:
    """Extract text from A2A JSON-RPC response."""
    if rpc_response.get("error"):
        error = rpc_response["error"]
        message = error.get("message", "Agent request failed") if isinstance(error, dict) else "Agent request failed"
        return f"Error: {message}"

    result = rpc_response.get("result", {})
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return "No result returned"

    texts = []
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("text"):
                texts.append(part["text"])

    return (" ".join(texts) if texts else "No text in response")[:4_000]


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

a2a_client = A2AClient()


def _configured_agent_urls() -> list[str]:
    return [url.strip().rstrip("/") for url in AGENT_URLS.split(",") if url.strip()]


async def _discover_missing_agents() -> None:
    for url in _configured_agent_urls():
        if not any(agent.url == url for agent in a2a_client.list_agents()):
            await a2a_client.discover(url)


async def _discovery_loop() -> None:
    while True:
        await _discover_missing_agents()
        await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Discover agents on startup."""
    urls = _configured_agent_urls()
    logger.info("Discovering %d agents...", len(urls))

    # Try discovery with retries for startup ordering
    for attempt in range(3):
        await _discover_missing_agents()

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
    discovery_task = asyncio.create_task(_discovery_loop())
    try:
        yield
    finally:
        discovery_task.cancel()
        try:
            await discovery_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Multi-Agent Health Workflow Demo Orchestrator",
    description=(
        "Routes synthetic scenarios through a small A2A 0.3-style teaching subset. "
        "This is not a protocol-conformance or medical system."
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


@app.get("/ready")
async def ready():
    expected = len(_configured_agent_urls())
    discovered = len(a2a_client.list_agents())
    if discovered < expected:
        raise HTTPException(
            status_code=503,
            detail=f"Waiting for agents: discovered {discovered} of {expected}",
        )
    return {
        "status": "ready",
        "agents_discovered": discovered,
        "agent_names": [agent.name for agent in a2a_client.list_agents()],
    }


@app.get("/api/v1/agents")
async def list_agents():
    """List all discovered agents."""
    agents = a2a_client.list_agents()
    return {
        "agents": [a.model_dump() for a in agents],
        "count": len(agents),
    }


@app.post("/api/v1/workflow", response_model=models.WorkflowResponse)
async def run_workflow(request: models.WorkflowRequest):
    """Execute a multi-agent workflow."""
    return await execute_workflow(
        a2a_client,
        query=request.query,
        workflow_type=request.workflow_type,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
