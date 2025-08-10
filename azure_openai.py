# azure_openai.py
import os
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# optional dynamic installer (keeps original idea but safe if pipmaster missing)
try:
    import pipmaster as pm  # type: ignore
except Exception:
    pm = None

if pm is not None:
    try:
        if not pm.is_installed("openai"):
            pm.install("openai")
        if not pm.is_installed("tenacity"):
            pm.install("tenacity")
    except Exception:
        # don't fail import if dynamic install fails; user likely has deps already
        pass

from openai import AsyncAzureOpenAI, APIConnectionError, RateLimitError, APITimeoutError  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # type: ignore

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
)

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_str(x: Any) -> str:
    """Convert value to str safely (None -> empty string)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _normalize_messages(
    system_prompt: Optional[str],
    history_messages: Optional[List[Union[Dict[str, Any], str]]],
    prompt: Optional[str],
) -> List[Dict[str, str]]:
    """
    Build and sanitize messages list for the chat API.
    Remove messages that have no content after stripping.
    Accepts history items as dicts {'role':..., 'content':...} or plain strings (assumed user).
    """
    msgs: List[Dict[str, str]] = []

    if system_prompt is not None:
        sp = _ensure_str(system_prompt).strip()
        if sp:
            msgs.append({"role": "system", "content": sp})
        else:
            logger.debug("System prompt was empty -> skipped")

    if history_messages:
        for i, item in enumerate(history_messages):
            try:
                if isinstance(item, str):
                    content = item.strip()
                    if content:
                        msgs.append({"role": "user", "content": content})
                    else:
                        logger.debug("Skipped empty history_messages[%d] (str)", i)
                elif isinstance(item, dict):
                    role = _ensure_str(item.get("role", "user")).strip() or "user"
                    content = _ensure_str(item.get("content", "")).strip()
                    if content:
                        msgs.append({"role": role, "content": content})
                    else:
                        logger.debug("Skipped empty history_messages[%d] (dict)", i)
                else:
                    # unknown type -> coerce to string
                    content = _ensure_str(item).strip()
                    if content:
                        msgs.append({"role": "user", "content": content})
                    else:
                        logger.debug("Skipped empty/unknown history_messages[%d]", i)
            except Exception:
                logger.exception("Error normalizing history_messages[%d], skipping", i)
                continue

    if prompt is not None:
        p = _ensure_str(prompt).strip()
        if p:
            msgs.append({"role": "user", "content": p})
        else:
            logger.debug("Prompt was empty -> skipped")

    # Final defensive filter (shouldn't be needed but keep safe)
    filtered = [m for m in msgs if isinstance(m, dict) and m.get("content") and str(m["content"]).strip()]
    if len(filtered) != len(msgs):
        logger.debug("Filtered out %d empty messages", len(msgs) - len(filtered))
    return filtered


# retry decorator for completions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def azure_openai_complete_if_cache(
    model: str,
    prompt: Optional[str],
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Union[Dict[str, Any], str]]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs,
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Call Azure OpenAI chat completion safely.
    Returns:
      - async generator of strings if streaming
      - plain string if non-streaming
    If all messages are empty (after normalization) this function returns an empty string
    and does NOT call the API (to avoid sending null content).
    """

    # set env vars if provided
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    # build client
    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=model,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    kwargs.pop("hashing_kv", None)

    # normalize and sanitize messages
    messages = _normalize_messages(system_prompt, history_messages, prompt)

    # if no useful messages -> short-circuit and avoid API call (prevents sending null)
    if not messages:
        logger.warning("No non-empty messages to send to the LLM; returning empty string to avoid API error.")
        return ""


    use_parse = "response_format" in kwargs
    try:
        if use_parse and hasattr(openai_async_client, "beta"):
            response = await openai_async_client.beta.chat.completions.parse(model=model, messages=messages, **kwargs)
        else:
            response = await openai_async_client.chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception:

        logger.exception("Error while calling Azure OpenAI chat completion")
        raise


    if hasattr(response, "__aiter__"):
        async def _stream_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in response:
                    # chunk may be object-like or dict-like - handle both
                    try:
                        choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else None)
                        if not choices:
                            continue
                        first = choices[0]

                        delta = getattr(first, "delta", None) or (first.get("delta") if isinstance(first, dict) else None)
                        content = None
                        if delta is not None:
                            content = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)
                        if content is None:

                            message_obj = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
                            if message_obj:
                                content = getattr(message_obj, "content", None) or (message_obj.get("content") if isinstance(message_obj, dict) else None)
                        if content is None:
                            # nothing useful in this chunk
                            continue
                        if isinstance(content, str) and r"\u" in content:
                            try:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            except Exception:
                                logger.debug("safe_unicode_decode failed for chunk content; using raw string")
                        yield _ensure_str(content)
                    except Exception:
                        logger.exception("Error processing stream chunk; skipping chunk")
                        continue
            except Exception:
                logger.exception("Error iterating over streaming response")
                raise

        return _stream_generator()


    content: Optional[str] = None
    try:

        choices = getattr(response, "choices", None) or (response.get("choices") if isinstance(response, dict) else None)
        if choices and len(choices) > 0:
            first = choices[0]

            message_obj = getattr(first, "message", None) or (first.get("message") if isinstance(first, dict) else None)
            if message_obj:
                content = getattr(message_obj, "content", None) or (message_obj.get("content") if isinstance(message_obj, dict) else None)

            if content is None:
                content = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
    except Exception:
        logger.exception("Error extracting content from non-streaming response")

    if content is None:
        logger.debug("Non-streaming response contained no content; returning empty string")
        return ""

    if isinstance(content, str) and r"\u" in content:
        try:
            content = safe_unicode_decode(content.encode("utf-8"))
        except Exception:
            logger.debug("safe_unicode_decode failed for non-streaming content; using raw string")

    return _ensure_str(content)


async def azure_openai_complete(
    prompt: Optional[str],
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Union[Dict[str, Any], str]]] = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Convenience wrapper: calls azure_openai_complete_if_cache using LLN_MODEL env var (or fallback).
    Returns either a string (non-stream) or an async generator (if the underlying call streams).
    If keyword_extraction is True, and the result is a string, it will try to extract JSON body.
    """
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    result = await azure_openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


    if hasattr(result, "__aiter__"):
        return result

    # result expected to be string now
    text = _ensure_str(result)
    if keyword_extraction:
        try:
            return locate_json_string_body_from_string(text)
        except Exception:
            logger.exception("Failed to extract JSON body from result; returning raw text")
            return text
    return text


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def azure_openai_embed(
    texts: List[Optional[str]],
    model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
) -> np.ndarray:
    """
    Safe wrapper for embeddings: preserves input list length (None -> empty string)
    and returns numpy array of embeddings.
    """
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=model,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


    sanitized = [_ensure_str(t) for t in texts]
    response = await openai_async_client.embeddings.create(model=model, input=sanitized, encoding_format="float")
    # response.data -> list of objects with .embedding or dicts with ['embedding']
    embeddings = []
    for dp in response.data:
        emb = getattr(dp, "embedding", None) or (dp.get("embedding") if isinstance(dp, dict) else None)
        if emb is None:
            logger.debug("Found a data point without embedding; using zeros")
            embeddings.append([0.0] * 1536)
        else:
            embeddings.append(emb)
    return np.array(embeddings)
