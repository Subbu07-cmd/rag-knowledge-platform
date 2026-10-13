"""Reusable LangGraph agent template.

Copy this file, implement the node functions, and you get a typed, testable agent
with a bounded tool-use loop and a final answer node.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    task: str
    steps: list[dict]
    observation: str
    answer: str
    iterations: int


def make_agent(
    plan: Callable[[AgentState], dict],          # -> {"tool": name, "input": ...} or {"final": text}
    tools: dict[str, Callable[[object], str]],
    max_iterations: int = 5,
):
    def think(state: AgentState) -> AgentState:
        decision = plan(state)
        steps = state.get("steps", []) + [decision]
        return {"steps": steps, "iterations": state.get("iterations", 0) + 1}

    def route(state: AgentState) -> str:
        last = state["steps"][-1]
        if "final" in last or state["iterations"] >= max_iterations:
            return "finish"
        return "act"

    def act(state: AgentState) -> AgentState:
        last = state["steps"][-1]
        tool = tools.get(last.get("tool", ""))
        obs = tool(last.get("input")) if tool else f"Unknown tool: {last.get('tool')}"
        return {"observation": obs}

    def finish(state: AgentState) -> AgentState:
        last = state["steps"][-1]
        return {"answer": last.get("final") or state.get("observation", "No answer within iteration budget.")}

    g = StateGraph(AgentState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("finish", finish)
    g.set_entry_point("think")
    g.add_conditional_edges("think", route, {"act": "act", "finish": "finish"})
    g.add_edge("act", "think")
    g.add_edge("finish", END)
    return g.compile()
