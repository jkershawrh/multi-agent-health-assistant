"""Gradio UI for the synthetic multi-agent health workflow demo."""

import os

import gradio as gr
import httpx

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")


def run_workflow(query: str) -> str:
    """POST an invented scenario to the orchestrator workflow endpoint."""
    if not query.strip():
        return "Please enter an invented scenario."
    try:
        resp = httpx.post(
            f"{ORCHESTRATOR_URL}/api/v1/workflow",
            json={"query": query, "workflow_type": "patient_triage"},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"Workflow request rejected (HTTP {exc.response.status_code})."
    except httpx.RequestError:
        return "Connection error: could not reach the orchestrator."
    except ValueError:
        return "Workflow response was not valid JSON."

    if not isinstance(data, dict):
        return "Workflow response had an unexpected shape."

    lines: list[str] = []
    lines.append(f"Workflow status: {data.get('status', 'unknown')}")
    if data.get("failed_step"):
        lines.append(f"Stopped at: {data['failed_step']}")
    lines.append("")
    for i, step in enumerate(data.get("steps", []), start=1):
        lines.append(f"--- Step {i} ---")
        lines.append(f"  Agent:   {step.get('agent', 'unknown')}")
        lines.append(f"  Action:  {step.get('action', 'N/A')}")
        lines.append(f"  Result:  {step.get('result', 'N/A')}")
        lines.append(f"  Latency: {step.get('latency_ms', 'N/A')} ms")
        lines.append("")

    total = data.get("total_latency_ms")
    if total is not None:
        lines.append(f"Total latency: {total} ms")

    lines.append("")
    lines.append(f"**Safety boundary:** {data.get('ai_disclaimer', 'Educational demo only.')}")
    return "\n".join(lines)


def fetch_agents() -> str:
    """GET the agent registry from the orchestrator."""
    try:
        resp = httpx.get(
            f"{ORCHESTRATOR_URL}/api/v1/agents",
            timeout=10.0,
        )
        resp.raise_for_status()
        agents = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"Agent request rejected (HTTP {exc.response.status_code})."
    except httpx.RequestError:
        return "Connection error: could not reach the orchestrator."
    except ValueError:
        return "Agent response was not valid JSON."

    agent_list = agents.get("agents", []) if isinstance(agents, dict) else []
    if not isinstance(agent_list, list) or not agent_list:
        return "No agents discovered."

    lines: list[str] = []
    for agent in agent_list:
        lines.append(f"Name:   {agent.get('name', 'unknown')}")
        lines.append(f"URL:    {agent.get('url', 'N/A')}")
        lines.append(f"Status: {agent.get('status', 'N/A')}")
        skills = agent.get("skills", [])
        if skills:
            skill_names = [
                s.get("name", s.get("id", str(s)))
                if isinstance(s, dict) else str(s)
                for s in skills
            ]
            lines.append(f"Skills: {', '.join(skill_names)}")
        lines.append("")
    return "\n".join(lines)


def fetch_stats() -> str:
    """GET health/statistics from the orchestrator."""
    try:
        resp = httpx.get(
            f"{ORCHESTRATOR_URL}/health",
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return f"Health request rejected (HTTP {exc.response.status_code})."
    except httpx.RequestError:
        return "Connection error: could not reach the orchestrator."
    except ValueError:
        return "Health response was not valid JSON."

    if not isinstance(data, dict):
        return "Health response had an unexpected shape."

    lines: list[str] = []
    lines.append(f"Status:      {data.get('status', 'N/A')}")

    agent_count = data.get("agents_discovered", "N/A")
    lines.append(f"Agent count: {agent_count}")

    agent_names = data.get("agent_names", [])
    if agent_names:
        lines.append(f"Agents:      {', '.join(str(n) for n in agent_names)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Multi-Agent Health Workflow Demo") as demo:
    gr.Markdown("# Multi-Agent Health Workflow Demo")
    gr.Markdown(
        "Use invented scenarios only. This interface demonstrates software routing; "
        "it does not provide care, triage, diagnosis, treatment, scheduling, or emergency help."
    )

    with gr.Tab("Synthetic Workflow"):
        query_input = gr.Textbox(
            label="Invented Scenario",
            placeholder="Example: Synthetic intake event for routing demonstration",
            lines=3,
            max_lines=10,
        )
        run_btn = gr.Button("Run Workflow")
        workflow_output = gr.Textbox(label="Workflow Results", lines=20)
        run_btn.click(fn=run_workflow, inputs=query_input, outputs=workflow_output)

    with gr.Tab("Agent Registry"):
        refresh_btn = gr.Button("Refresh")
        agents_output = gr.Textbox(label="Discovered Agents", lines=15)
        refresh_btn.click(fn=fetch_agents, inputs=[], outputs=agents_output)

    with gr.Tab("Statistics"):
        stats_btn = gr.Button("Refresh")
        stats_output = gr.Textbox(label="System Info", lines=10)
        stats_btn.click(fn=fetch_stats, inputs=[], outputs=stats_output)

    gr.Markdown(
        "---\n"
        "⚠️ Educational simulation only. Do not enter personal or health information. "
        "This demo is not medical advice or an emergency service."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
