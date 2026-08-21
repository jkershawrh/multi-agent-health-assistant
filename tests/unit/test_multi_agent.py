"""Stage 2: unit validation for the A2A-style teaching subset."""

import asyncio
import pathlib
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

import agent
import models
import orchestrator


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _message_request(task_id: str = "task-001") -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": "message/send",
        "params": {
            "id": task_id,
            "message": {
                "messageId": "message-1",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Synthetic intake event for routing demonstration",
                    }
                ],
            },
        },
    }


@pytest.fixture
def triage_client(monkeypatch):
    monkeypatch.setattr(agent, "AGENT_NAME", "triage")
    monkeypatch.setattr(agent, "AGENT_PORT", 8001)
    monkeypatch.setattr(agent, "AGENT_SKILLS_RAW", "classify,prioritize")
    monkeypatch.setattr(agent, "MODEL_ENDPOINT", "")
    monkeypatch.setattr(agent, "DEMO_MODE", True)
    agent.TASKS.clear()
    return TestClient(agent.app)


class TestAgentCard:
    def test_card_identifies_the_versioned_teaching_shape(self, triage_client):
        response = triage_client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        card = response.json()
        assert card["name"] == "triage"
        assert card["protocolVersion"] == "0.3.0"
        assert card["preferredTransport"] == "JSONRPC"
        assert card["url"].endswith("/a2a")
        assert card["defaultInputModes"] == ["text/plain"]
        assert len(card["skills"]) == 2

    def test_model_defaults_are_not_shared(self):
        first = models.AgentCard(name="one", description="First", skills=[])
        second = models.AgentCard(name="two", description="Second", skills=[])

        first.skills.append(
            models.AgentSkill(id="demo", name="Demo", description="Demo skill")
        )

        assert second.skills == []


class TestJsonRpcTasks:
    def test_message_send_returns_labeled_simulation(self, triage_client):
        response = triage_client.post("/a2a", json=_message_request())

        assert response.status_code == 200
        payload = response.json()
        assert "error" not in payload
        task = payload["result"]
        artifact = task["artifacts"][0]
        text = artifact["parts"][0]["text"]
        assert task["status"]["state"] == "completed"
        assert "DEMO SIMULATION" in text
        assert "NO TRIAGE DECISION" in text
        assert artifact["metadata"]["source"] == "demo-simulator"
        assert artifact["metadata"]["simulation"] is True

    def test_tasks_get_returns_a_stored_task(self, triage_client):
        triage_client.post("/a2a", json=_message_request(task_id="stored-task"))

        response = triage_client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "request-2",
                "method": "tasks/get",
                "params": {"id": "stored-task"},
            },
        )

        assert response.json()["result"]["id"] == "stored-task"

    def test_tasks_get_rejects_unknown_task(self, triage_client):
        response = triage_client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "request-3",
                "method": "tasks/get",
                "params": {"id": "missing"},
            },
        )

        assert response.json()["error"]["code"] == -32001

    def test_tasks_get_rejects_invalid_task_id(self, triage_client):
        response = triage_client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "request-invalid-id",
                "method": "tasks/get",
                "params": {"id": ["not", "a", "string"]},
            },
        )

        assert response.json()["error"]["code"] == -32602

    def test_invalid_message_returns_json_rpc_error(self, triage_client):
        request = _message_request()
        request["params"]["message"]["parts"] = []

        response = triage_client.post("/a2a", json=request)

        assert response.json()["error"]["code"] == -32602

    def test_unknown_method_returns_json_rpc_error(self, triage_client):
        response = triage_client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": "request-4", "method": "unknown"},
        )

        assert response.json()["error"]["code"] == -32601

    def test_live_model_failure_never_substitutes_demo_output(
        self,
        triage_client,
        monkeypatch,
    ):
        async def unavailable(_text: str) -> str:
            raise httpx.ConnectError("unavailable")

        monkeypatch.setattr(agent, "MODEL_ENDPOINT", "https://models.example.test/v1")
        monkeypatch.setattr(agent, "DEMO_MODE", False)
        monkeypatch.setattr(agent, "_llm_response", unavailable)

        response = triage_client.post("/a2a", json=_message_request())

        payload = response.json()
        assert payload["error"]["code"] == -32001
        assert "no demo result was substituted" in payload["error"]["message"]


class FakeWorkflowClient:
    def __init__(self, fail_at: str | None = None):
        self.fail_at = fail_at
        self.calls: list[tuple[str, str]] = []

    async def send_task(self, agent_name: str, text: str) -> dict:
        self.calls.append((agent_name, text))
        if agent_name == self.fail_at:
            return {"error": {"message": "Agent request failed"}}
        return {
            "result": {
                "artifacts": [
                    {
                        "parts": [
                            {"text": f"[DEMO SIMULATION] {agent_name} handoff complete"}
                        ]
                    }
                ]
            }
        }


class TestWorkflowExecution:
    def test_workflow_executes_all_steps_and_passes_context(self):
        client = FakeWorkflowClient()

        result = _run(
            orchestrator.execute_workflow(
                client,
                "Synthetic intake event",
                "patient_triage",
            )
        )

        assert result.status == "completed"
        assert result.agents_involved == ["triage", "clinical", "scheduling"]
        assert [step.action for step in result.steps] == [
            "classify",
            "diagnose",
            "schedule",
        ]
        assert "Previous step (triage/classify)" in client.calls[1][1]

    def test_workflow_stops_after_an_agent_failure(self):
        client = FakeWorkflowClient(fail_at="clinical")

        result = _run(
            orchestrator.execute_workflow(client, "Synthetic event", "patient_triage")
        )

        assert result.status == "failed"
        assert result.failed_step == "clinical/diagnose"
        assert [call[0] for call in client.calls] == ["triage", "clinical"]

    def test_client_rejects_malformed_agent_response(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": "response-with-no-result"},
            )

        client = orchestrator.A2AClient(transport=httpx.MockTransport(handler))
        client.agents["triage"] = models.DiscoveredAgent(
            name="triage",
            url="http://triage-agent:8001",
        )

        result = _run(client.send_task("triage", "Synthetic event"))

        assert result == {"error": "Agent returned an invalid response"}


class TestValidationAndHealth:
    def test_workflow_request_rejects_unknown_type_and_oversized_query(self):
        with pytest.raises(ValidationError):
            models.WorkflowRequest(query="synthetic", workflow_type="unknown")
        with pytest.raises(ValidationError):
            models.WorkflowRequest(query="x" * 2_001)

    def test_demo_health_and_readiness_are_explicit(self, triage_client):
        health = triage_client.get("/health")
        ready = triage_client.get("/ready")

        assert health.json()["mode"] == "demo"
        assert health.json()["model"] == "demo-simulator"
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
