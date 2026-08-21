"""Stage 3: route a synthetic workflow through three in-process agent APIs."""

import asyncio
import pathlib
import sys

import httpx
from fastapi.testclient import TestClient

SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

import agent
import orchestrator

AGENT_PORTS = {"triage": 8001, "clinical": 8002, "scheduling": 8003}


class AgentRouterTransport(httpx.AsyncBaseTransport):
    """Route fake service hostnames to independently configured agent roles."""

    def __init__(self):
        self.transport = httpx.ASGITransport(app=agent.app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        name = request.url.host.removesuffix("-agent")
        agent.AGENT_NAME = name
        agent.AGENT_PORT = AGENT_PORTS[name]
        agent.AGENT_SKILLS_RAW = {
            "triage": "classify,prioritize",
            "clinical": "diagnose,recommend",
            "scheduling": "schedule,notify",
        }[name]
        return await self.transport.handle_async_request(request)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _discovered_client() -> orchestrator.A2AClient:
    client = orchestrator.A2AClient(transport=AgentRouterTransport())
    for name, port in AGENT_PORTS.items():
        discovered = await client.discover(f"http://{name}-agent:{port}")
        assert discovered is not None
    return client


def test_workflow_api_routes_all_three_steps(monkeypatch):
    monkeypatch.setattr(agent, "MODEL_ENDPOINT", "")
    monkeypatch.setattr(agent, "DEMO_MODE", True)
    agent.TASKS.clear()
    routed_client = _run(_discovered_client())
    monkeypatch.setattr(orchestrator, "a2a_client", routed_client)

    with TestClient(orchestrator.app) as client:
        ready = client.get("/ready")
        response = client.post(
            "/api/v1/workflow",
            json={
                "query": "Synthetic intake event for routing demonstration",
                "workflow_type": "patient_triage",
            },
        )

    assert ready.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed", payload["steps"][0]["result"]
    assert payload["agents_involved"] == ["triage", "clinical", "scheduling"]
    assert len(payload["steps"]) == 3
    for step in payload["steps"]:
        assert "DEMO SIMULATION" in step["result"]
        assert step["latency_ms"] >= 0


def test_workflow_api_rejects_non_synthetic_shape(monkeypatch):
    routed_client = _run(_discovered_client())
    monkeypatch.setattr(orchestrator, "a2a_client", routed_client)

    with TestClient(orchestrator.app) as client:
        response = client.post(
            "/api/v1/workflow",
            json={"query": "", "workflow_type": "unsupported"},
        )

    assert response.status_code == 422
