"""Planner -> writer -> verifier graph (LangGraph).

    plan ──> write ──> verify ──┬──> write   (steps that failed, with the verifier's feedback)
                                └──> END     (all pass, or MAX_VERIFY_ROUNDS reached: flag the rest)

The verifier checks every number against telemetry and the reference hard limits.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from sim2sop import config
from sim2sop.agent.planner import PlanResult
from sim2sop.agent.verifier import verify_step
from sim2sop.profiles import Profile
from sim2sop.schema import Flag, Step


class GraphState(TypedDict, total=False):
    steps: list[Step]
    pending: list[str]                  # step ids to (re)write
    feedback: dict[str, list[str]]      # step id -> verifier errors from the last round
    round: int
    flags: list[Flag]
    anchor_run: str | None


def build_graph(profile: Profile, writer: Any, planner: Callable[[], PlanResult],
                max_rounds: int | None = None):
    """`planner` runs the planner with the run sets / limits chosen by the caller."""
    max_rounds = config.MAX_VERIFY_ROUNDS if max_rounds is None else max_rounds

    def plan_node(state: GraphState) -> GraphState:
        result = planner()
        return {"steps": result.steps, "pending": result.dirty, "feedback": {}, "round": 0,
                "flags": result.flags, "anchor_run": result.anchor_run}

    def write_node(state: GraphState) -> GraphState:
        by_id = {s.id: s for s in state["steps"]}
        for sid in state["pending"]:
            step = by_id[sid]
            text = writer.write(step, state["feedback"].get(sid))
            for field, value in text.items():
                setattr(step, field, value if value is not None else ("" if field != "intent" else None))
        return {"steps": state["steps"]}

    def verify_node(state: GraphState) -> GraphState:
        by_id = {s.id: s for s in state["steps"]}
        failed: dict[str, list[str]] = {}
        flags = list(state["flags"])
        last_round = state["round"] + 1 > max_rounds
        for sid in state["pending"]:
            step = by_id[sid]
            verdict = verify_step(step, profile)
            step.verified = verdict.ok
            step.flags = [f for f in step.flags if not f.startswith(("Unverified", "Callout"))]
            step.flags += verdict.warnings
            if verdict.ok:
                continue
            if last_round:
                step.flags += [f"Unverified — {e}" for e in verdict.errors]
                flags += [Flag(kind="verifier", step_id=sid, message=e) for e in verdict.errors]
            else:
                failed[sid] = verdict.errors
        return {"pending": list(failed), "feedback": failed, "round": state["round"] + 1, "flags": flags}

    def route(state: GraphState) -> str:
        return "write" if state["pending"] else END

    g = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("write", write_node)
    g.add_node("verify", verify_node)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route, {"write": "write", END: END})
    g.add_edge("write", "verify")
    g.add_conditional_edges("verify", route, {"write": "write", END: END})
    return g.compile()
