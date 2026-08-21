"""Stage 4: lightweight performance guards for the in-process demo path."""

import asyncio
import json
import pathlib
import sys
import time

import httpx
from fastapi.testclient import TestClient

SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

import agent  # noqa: E402
import models  # noqa: E402
import orchestrator  # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class FastWorkflowClient:
    async def send_task(self, agent_name: str, _text: str) -> dict:
        return {
            "result": {
                "artifacts": [
                    {"parts": [{"text": f"[DEMO SIMULATION] {agent_name} handoff"}]}
                ]
            }
        }


def test_three_step_demo_workflow_is_below_five_seconds():
    start = time.perf_counter()
    response = _run(
        orchestrator.execute_workflow(
            FastWorkflowClient(),
            "Synthetic event",
            "patient_triage",
        )
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status == "completed"
    assert len(response.steps) == 3
    assert elapsed_ms < 5_000


def test_three_agent_discovery_is_below_two_seconds():
    card = models.AgentCard(
        name="demo",
        description="Demo teaching agent",
        skills=[models.AgentSkill(id="demo", name="Demo", description="Demo skill")],
    ).model_dump()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = dict(card)
        payload["name"] = request.url.host.split("-")[0]
        return httpx.Response(200, content=json.dumps(payload))

    async def discover_all() -> orchestrator.A2AClient:
        client = orchestrator.A2AClient(
            transport=httpx.MockTransport(handler),
        )
        for name, port in (("triage", 8001), ("clinical", 8002), ("scheduling", 8003)):
            await client.discover(f"http://{name}-agent:{port}")
        return client

    start = time.perf_counter()
    client = _run(discover_all())
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(client.list_agents()) == 3
    assert elapsed_ms < 2_000


def test_individual_demo_agent_response_is_below_one_second(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_NAME", "triage")
    monkeypatch.setattr(agent, "MODEL_ENDPOINT", "")
    monkeypatch.setattr(agent, "DEMO_MODE", True)
    agent.TASKS.clear()
    client = TestClient(agent.app)
    request = {
        "jsonrpc": "2.0",
        "id": "benchmark",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Synthetic benchmark event"}],
            }
        },
    }

    start = time.perf_counter()
    response = client.post("/a2a", json=request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert response.status_code == 200
    assert response.json()["result"]["status"]["state"] == "completed"
    assert elapsed_ms < 1_000
