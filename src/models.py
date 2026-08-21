"""Pydantic models for the multi-agent health assistant.

Defines the small A2A 0.3-style teaching subset used for agent discovery,
JSON-RPC task operations, and workflow orchestration.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# A2A Protocol models
# ---------------------------------------------------------------------------


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = True


class AgentSkill(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    tags: list[str] | None = Field(default=None, max_length=20)
    examples: list[str] | None = Field(default=None, max_length=10)


class AgentCard(BaseModel):
    name: str
    description: str
    version: str = "0.1.0"
    url: str = "http://localhost:8001"
    protocolVersion: str = "0.3.0"
    preferredTransport: str = "JSONRPC"
    provider: str = "Red Hat / Intel"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    skills: list[AgentSkill] = Field(default_factory=list)


class Part(BaseModel):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=2_000)


class Message(BaseModel):
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: Literal["message"] = "message"
    role: Literal["user"] = "user"
    parts: list[Part] = Field(min_length=1, max_length=10)


class TaskStatus(BaseModel):
    state: Literal["submitted", "working", "completed", "failed"] = "submitted"
    timestamp: str | None = None


class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parts: list[Part] = Field(min_length=1, max_length=10)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=128)
    contextId: str | None = None
    status: TaskStatus = Field(default_factory=TaskStatus)
    artifacts: list[Artifact] | None = None
    kind: Literal["task"] = "task"


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=128,
    )
    method: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Task | None = None
    error: dict | None = None

    @model_validator(mode="after")
    def require_result_or_error(self):
        if (self.result is None) == (self.error is None):
            raise ValueError("JSON-RPC response must contain exactly one of result or error")
        return self


# ---------------------------------------------------------------------------
# Orchestrator models
# ---------------------------------------------------------------------------


class DiscoveredAgent(BaseModel):
    name: str
    url: str
    status: str = "active"
    skills: list[AgentSkill] = Field(default_factory=list)


class WorkflowRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    workflow_type: Literal["patient_triage", "general"] = "general"


class WorkflowStep(BaseModel):
    agent: str
    action: str
    result: str
    latency_ms: float


class WorkflowResponse(BaseModel):
    steps: list[WorkflowStep]
    total_latency_ms: float
    agents_involved: list[str]
    status: Literal["completed", "failed"] = "completed"
    failed_step: str | None = None
    ai_disclaimer: str = (
        "Educational simulation only. It does not provide medical advice, diagnosis, "
        "treatment, triage, scheduling, or emergency services. Use synthetic data only."
    )
