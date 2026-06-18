"""Shared LangChain ChatOpenAI factory for pipeline LLM calls."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .inference_config import llm_model_name, openai_base


def make_chat_llm(
    *,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> ChatOpenAI:
    """OpenAI-compatible chat client using GT_OPENAI_BASE / GT_LLM_MODEL."""
    return ChatOpenAI(
        model=llm_model_name(),
        openai_api_key="EMPTY",
        openai_api_base=openai_base(),
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
