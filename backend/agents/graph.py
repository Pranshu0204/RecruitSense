"""LangGraph orchestration for the screening pipeline.

Topology::

           ┌─────► rag ─────┐
    parse ─┤                ├──► score ──► END
           └─────► bias ────┘
"""

import asyncio
from typing import TypedDict

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


def build_graph():
    """Compile the LangGraph DAG (cached)."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    g: StateGraph = StateGraph(GraphState)
    g.add_node("parse", _parse_node)
    g.add_node("rag", _rag_node)
    g.add_node("bias", _bias_node)
    g.add_node("score", _score_node)

    g.set_entry_point("parse")
    g.add_edge("parse", "rag")
    g.add_edge("parse", "bias")
    g.add_edge("rag", "score")
    g.add_edge("bias", "score")
    g.add_edge("score", END)

    _compiled_graph = g.compile()
    return _compiled_graph


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
