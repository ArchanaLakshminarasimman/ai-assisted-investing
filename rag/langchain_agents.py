from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


DEFAULT_LANGCHAIN_MODEL = "gpt-4.1-mini"


class AnalystDraft(BaseModel):
    draft_insight: str = Field(
        description=(
            "One short paragraph explaining the stock ranking with at least "
            "one upside driver and one risk."
        )
    )
    key_points: list[str] = Field(
        description="One to three evidence-backed supporting points.",
    )
    risk_points: list[str] = Field(
        description="One or two evidence-backed risk points.",
    )
    evidence_titles_used: list[str] = Field(
        description="Titles of the evidence items used in the draft.",
    )


class ReviewerDecision(BaseModel):
    final_insight: str = Field(
        description=(
            "Cleaned final paragraph that keeps only claims supported by the evidence."
        )
    )
    key_points: list[str] = Field(
        description="One to three verified supporting points.",
    )
    risk_points: list[str] = Field(
        description="One or two verified risk points.",
    )
    unsupported_claims: list[str] = Field(
        description=(
            "Claims from the analyst draft that were not fully supported by the evidence."
        )
    )
    confidence_score: float = Field(
        description="Confidence from 0.0 to 1.0 based on evidence support."
    )
    review_summary: str = Field(
        description="One sentence summarizing the review outcome.",
    )


@dataclass
class LangChainInsightResult:
    insight: str
    key_points: list[str]
    risk_points: list[str]
    confidence_score: float | None = None
    unsupported_claims: list[str] | None = None
    review_summary: str | None = None


def _require_openai_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The LangChain generator currently uses "
            "ChatOpenAI under the hood."
        )


def _normalize_model_name(model: str | None) -> str:
    model_name = (
        model
        or os.environ.get("LANGCHAIN_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_LANGCHAIN_MODEL
    )
    if model_name.startswith("openai:"):
        return model_name.split(":", maxsplit=1)[1]

    return model_name


def _analyst_system_prompt() -> str:
    return (
        "You are the analyst agent in an equity research workflow. "
        "Use only the ranking fields and retrieved evidence provided in the user payload. "
        "Do not invent facts, do not rely on outside knowledge, and keep the explanation concise. "
        "Your job is to draft the best possible evidence-backed investment explanation."
    )


def _reviewer_system_prompt() -> str:
    return (
        "You are the reviewer agent in an equity research workflow. "
        "Your job is to check the analyst draft against the supplied evidence, "
        "remove or soften unsupported claims, and return a cleaner final explanation. "
        "Use only the provided payload, do not invent facts, and assign a confidence "
        "score from 0.0 to 1.0 based on evidence quality and consistency."
    )


@lru_cache(maxsize=4)
def _build_agents(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0)
    analyst_agent = create_agent(
        model=llm,
        system_prompt=_analyst_system_prompt(),
        response_format=AnalystDraft,
    )
    reviewer_agent = create_agent(
        model=llm,
        system_prompt=_reviewer_system_prompt(),
        response_format=ReviewerDecision,
    )
    return analyst_agent, reviewer_agent


def _invoke_structured_agent(agent: Any, payload: dict[str, Any]) -> BaseModel:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(payload, indent=2),
                }
            ]
        }
    )
    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError("LangChain agent did not return structured output.")

    return structured


def _dedupe_non_empty(items: list[str], limit: int) -> list[str]:
    unique_items: list[str] = []
    seen: set[str] = set()

    for item in items:
        normalized = " ".join(str(item).split()).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        unique_items.append(normalized)
        if len(unique_items) >= limit:
            break

    return unique_items


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def generate_langchain_insight(
    payload: dict[str, Any],
    model: str | None = None,
) -> LangChainInsightResult:
    _require_openai_api_key()

    model_name = _normalize_model_name(model)
    analyst_agent, reviewer_agent = _build_agents(model_name)

    analyst_output = _invoke_structured_agent(analyst_agent, payload)
    reviewer_payload = {
        "ranking_payload": payload,
        "analyst_draft": analyst_output.model_dump(),
    }
    reviewer_output = _invoke_structured_agent(reviewer_agent, reviewer_payload)

    key_points = _dedupe_non_empty(reviewer_output.key_points, limit=3)
    risk_points = _dedupe_non_empty(reviewer_output.risk_points, limit=2)
    unsupported_claims = _dedupe_non_empty(
        reviewer_output.unsupported_claims,
        limit=5,
    )

    return LangChainInsightResult(
        insight=reviewer_output.final_insight.strip(),
        key_points=key_points,
        risk_points=risk_points,
        confidence_score=_clamp_confidence(reviewer_output.confidence_score),
        unsupported_claims=unsupported_claims,
        review_summary=reviewer_output.review_summary.strip(),
    )
