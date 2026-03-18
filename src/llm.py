"""Shared LLM client for Azure AI Foundry (v1 API).

Uses AsyncOpenAI with base_url (recommended pattern, no api_version needed).
Works with any model deployed in Foundry: GPT, DeepSeek, Llama, etc.

Required env vars:
    AZURE_OPENAI_BASE_URL     - e.g. https://<resource>.openai.azure.com/openai/v1/
    AZURE_INFERENCE_CREDENTIAL - API key for the Foundry resource
    LLM_MODEL                  - deployment name (e.g. gpt-4.1, DeepSeek-V3.1)
    LLM_MAX_CONCURRENT         - max parallel calls (default 10)
"""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None
_semaphore = None


def get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=os.environ["AZURE_OPENAI_BASE_URL"],
            api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
        )
    return _client


def get_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(
            int(os.environ.get("LLM_MAX_CONCURRENT", "10"))
        )
    return _semaphore


def get_model():
    return os.environ.get("LLM_MODEL", "gpt-4.1")


async def call(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 4096,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """Call LLM via Azure Foundry v1 API with retry, timeout, and rate limiting."""
    client = get_client()
    semaphore = get_semaphore()
    model = get_model()
    last_error = None

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_completion_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


async def call_messages(
    messages: list[dict],
    max_tokens: int = 4096,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """Call LLM with a multi-turn message array."""
    client = get_client()
    semaphore = get_semaphore()
    model = get_model()
    last_error = None

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


def extract_json(response: str) -> dict | None:
    """Extract JSON from LLM response, handling code blocks and truncation."""
    # 1. JSON in code block
    m = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Direct JSON in text
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Truncated JSON (response cut by max_tokens)
    brace = response.find('{')
    if brace >= 0:
        fragment = response[brace:]
        for suffix in (']]}', ']}', '}'):
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                continue

    return None
