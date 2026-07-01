"""LangGraph orchestration for the screening pipeline.

Topology::

           ┌─────► rag ─────┐
    parse ─┤                ├──► score ──► END
           └─────► bias ────┘

Two compiled graphs are exposed:
  build_graph()         — no checkpointer, used by /screen and /batch
  build_session_graph() — MemorySaver checkpointer, used by /session routes
                          to persist state across Turn 1 → Turn 2 → Turn 3
"""

import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.agents.bias_agent import detect_bias_signals
from backend.agents.parser_agent import parse_resume
from backend.agents.rag_agent import retrieve_context
from backend.agents.scorer_agent import score_resume
from backend.core.schemas import JDInput, ParsedResume, ScoreOutput
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class GraphState(TypedDict, total=False):
    """Shared state threaded through each pipeline node."""

    jd: JDInput
    resume_text: str
    parsed_resume: ParsedResume
    rag_context: str
    bias_flags: list[str]
    score: ScoreOutput


# --- Nodes --------------------------------------------------------------------
async def _parse_node(state: GraphState) -> dict[str, ParsedResume]:
    parsed = await parse_resume(state["resume_text"])
    return {"parsed_resume": parsed}


async def _rag_node(state: GraphState) -> dict[str, str]:
    context, _ = await retrieve_context(state["jd"], state.get("parsed_resume"))
    return {"rag_context": context}


async def _bias_node(state: GraphState) -> dict[str, list[str]]:
    return {"bias_flags": detect_bias_signals(state["resume_text"])}


async def _score_node(state: GraphState) -> dict[str, ScoreOutput]:
    score = await score_resume(
        jd=state["jd"],
        parsed_resume=state["parsed_resume"],
        rag_context=state.get("rag_context", ""),
        bias_flags=state.get("bias_flags", []),
    )
    return {"score": score}


# --- Graph builder ------------------------------------------------------------
_compiled_graph = None
_session_graph = None
_session_checkpointer = MemorySaver()


def _build_state_graph() -> StateGraph:
    """Construct the shared StateGraph topology (nodes + edges)."""
    g: StateGraph = StateGraph(GraphState)
    g.add_node("parse", _parse_node)
    g.add_node("rag", _rag_node)
    g.add_node("bias", _bias_node)
    # Node id must not collide with the "score" state key, so name it "scorer".
    g.add_node("scorer", _score_node)
    g.set_entry_point("parse")
    g.add_edge("parse", "rag")
    g.add_edge("parse", "bias")
    g.add_edge("rag", "scorer")
    g.add_edge("bias", "scorer")
    g.add_edge("scorer", END)
    return g


def build_graph():
    """Compile the LangGraph DAG without checkpointing (used by /screen and /batch)."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph
    _compiled_graph = _build_state_graph().compile()
    return _compiled_graph


def build_session_graph():
    """Compile the LangGraph DAG with MemorySaver checkpointing (used by /session routes).

    The checkpointer persists GraphState after every node so that subsequent
    requests can load it by thread_id — enabling Turn 2 (reweight without a
    new LLM call) and Turn 3 (side-by-side comparison of two saved sessions).
    """
    global _session_graph
    if _session_graph is not None:
        return _session_graph
    _session_graph = _build_state_graph().compile(checkpointer=_session_checkpointer)
    return _session_graph


# --- Direct-call entry point (preferred for batch fan-out) -------------------
async def run_pipeline(jd: JDInput, resume_text: str, model: str = "") -> ScoreOutput:
    """Run the full screening pipeline for a single resume.

    Calls agents directly (bypassing LangGraph graph overhead) so it works
    efficiently in both single-screen and batch fan-out contexts.
    """
    parsed = await parse_resume(resume_text)
    rag_task = asyncio.create_task(retrieve_context(jd, parsed))
    bias_flags = detect_bias_signals(resume_text)
    rag_context, _ = await rag_task
    return await score_resume(
        jd=jd,
        parsed_resume=parsed,
        rag_context=rag_context,
        bias_flags=bias_flags,
        model=model,
    )
