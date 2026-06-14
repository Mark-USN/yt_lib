"""Utilities for converting MCP/FastMCP prompt messages to LLM input messages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LlmMessage:
    """ Provider-neutral text message used between MCP prompts and LLM clients."""

    role: str
    content: str


def coerce_content_to_text(content: Any) -> str:
    """ Coerce common MCP/FastMCP content shapes to plain text.
        Args:
            content: The content to coerce, which may be a string, an object with a string
                     ``.text`` attribute, or an iterable of such objects.
            returns:
                A plain string representing the content.

        Supports:
        - plain strings
        - objects with a string ``.text`` attribute, such as TextContent-like objects
        - iterables of strings and/or objects with a string ``.text`` attribute
    """
    if isinstance(content, str):
        return content

    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(content, Iterable):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue

            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)

        if parts:
            return "\n".join(parts)

    raise TypeError(f"Unsupported LLM content type: {type(content)!r}")


def read_field(obj: Any, name: str) -> Any:
    """ Read ``name`` from either a mapping or an attribute-based object.
        Args:
            obj: The object to read from, which may be a mapping or an attribute-based object
            name: The name of the field to read
        Returns:
            The value of the field
    """
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def prompt_result_messages_to_llm(messages: str | Iterable[Any]) -> list[LlmMessage]:
    """ Normalize FastMCP ``PromptResult.messages`` to ``list[LlmMessage]``.
        Args:
            messages: The messages to normalize, which may be a string or an iterable of message-like objects
        Returns:
            A list of ``LlmMessage`` objects

        FastMCP prompt results may be a single string or a sequence of message-like
        objects. Message-like objects may be mapping-based or attribute-based.
    """
    if isinstance(messages, str):
        return [LlmMessage(role="user", content=messages)]

    if not isinstance(messages, Iterable):
        raise TypeError(f"Expected messages to be str or iterable, got {type(messages)!r}")

    return [
        LlmMessage(
            role=str(read_field(message, "role")),
            content=coerce_content_to_text(read_field(message, "content")),
        )
        for message in messages
    ]


def llm_messages_to_openai_chat(messages: Sequence[LlmMessage]) -> list[dict[str, str]]:
    """ Convert provider-neutral messages to Chat Completions-style OpenAI messages.
        Args:
            messages: The messages to convert, which should be a sequence of ``LlmMessage`` objects
        Returns:
            A list of dictionaries representing OpenAI Chat Completions-style messages
    """
    return [{"role": message.role, "content": message.content} for message in messages]


def mcp_messages_to_openai_chat(messages: Iterable[Any]) -> list[dict[str, str]]:
    """ Convert MCP/FastMCP messages directly to Chat Completions-style messages.
        Args:
            messages: The messages to convert, which may be a string or an iterable of
                      message-like objects
        Returns:
            A list of dictionaries representing OpenAI Chat Completions-style messages
    """
    return llm_messages_to_openai_chat(prompt_result_messages_to_llm(messages))


def llm_messages_to_openai_responses_input(
    messages: Sequence[LlmMessage],
) -> list[dict[str, Any]]:
    """ Convert provider-neutral messages to OpenAI Responses API input items.
        Args:
            messages: The messages to convert, which should be a sequence of ``LlmMessage`` objects
        Returns:
            A list of dictionaries representing OpenAI Responses API input items
    """
    return [
        {
            "role": message.role,
            "content": [{"type": "input_text", "text": message.content}],
        }
        for message in messages
    ]
