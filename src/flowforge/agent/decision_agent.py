"""Single-shot supply-chain decisions. There is no chat history."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from flowforge.agent.tools import DECISION_TOOLS

SYSTEM_INSTRUCTION = (
    "You are FlowForge's supply-chain decision layer.\n"
    "You analyze the operational scored orders: the latest chronological period, "
    "scored by the trained late-delivery model.\n"
    "All quantitative claims must come from tool outputs.\n"
    "expected_revenue_exposure is a prioritization proxy, never a financial loss.\n"
    "Never invent metrics, probabilities, counts, costs, or operational facts.\n"
    "Use tools before making quantitative recommendations."
)

_DEFAULT_MODEL = "gpt-4.1-mini"
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
_PLACEHOLDER_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"

_ACTION_INSTRUCTIONS = {
    "analyze_hotspots": (
        "Analyze late-delivery risk hotspots on the operational scored orders. "
        "Call network_summary and risk_hotspots before you answer. "
        "Regions are ranked by total expected_revenue_exposure. "
        "Identify which regions deserve attention first. "
        "Keep the briefing short and executive. "
        "Every number in the response must be copied from a tool result."
    ),
    "prioritize_interventions": (
        "Prioritize which operational scored orders should be reviewed first. "
        "Call priority_orders and network_summary before you answer. "
        "Rank attention with expected_revenue_exposure. "
        "Treat that value only as a prioritization proxy, never as financial "
        "loss, cost, or margin impact. "
        "Keep the briefing short and executive. "
        "Every number in the response must be copied from a tool result."
    ),
    "compare_shipping_modes": (
        "Compare shipping modes on the operational scored orders. "
        "Call compare_shipping_modes before you answer. "
        "Call network_summary if overall context would change the recommendation. "
        "Say where operational attention should go by mode. "
        "Keep the briefing short and executive. "
        "Every number in the response must be copied from a tool result."
    ),
}


class DecisionBrief(BaseModel):
    """Structured executive brief returned by one decision action."""

    headline: str = Field(description="One sentence an operator can act on.")
    summary: str = Field(
        description="Two or three sentences. Use only facts present in tool results."
    )
    recommendations: list[str] = Field(
        description="A few concrete next actions. Do not invent quantities."
    )
    supporting_metrics: list[str] = Field(
        description=(
            "Short lines copied from tool results. Each line names the metric "
            "and includes the number returned by the tool."
        )
    )


def _require_api_key() -> str:
    # Variables already set in the environment take precedence over .env.
    load_dotenv(_ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == _PLACEHOLDER_KEY:
        raise RuntimeError(
            f"OPENAI_API_KEY is missing. Paste your key into {_ENV_PATH} "
            "or set it in the environment before calling run_decision_action."
        )
    return api_key


def _as_brief(result: dict) -> dict:
    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError(
            "The decision agent finished without a structured response. Retry the action."
        )
    if isinstance(structured, DecisionBrief):
        return structured.model_dump()
    if isinstance(structured, dict):
        return DecisionBrief.model_validate(structured).model_dump()
    raise RuntimeError(f"Unexpected structured response type: {type(structured).__name__}.")


def run_decision_action(action: str) -> dict:
    """Run one fixed decision action and return an executive brief.

    Supported actions are ``analyze_hotspots``, ``prioritize_interventions``,
    and ``compare_shipping_modes``. Each call is independent.
    """
    instruction = _ACTION_INSTRUCTIONS.get(action)
    if instruction is None:
        supported = ", ".join(_ACTION_INSTRUCTIONS)
        raise ValueError(f"Unknown action {action!r}. Supported actions: {supported}.")

    api_key = _require_api_key()
    model_name = os.environ.get("FLOWFORGE_OPENAI_MODEL", "").strip() or _DEFAULT_MODEL
    model = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    agent = create_agent(
        model=model,
        tools=DECISION_TOOLS,
        system_prompt=SYSTEM_INSTRUCTION,
        response_format=ToolStrategy(DecisionBrief),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": instruction}]},
        {"recursion_limit": 12},
    )
    return _as_brief(result)
