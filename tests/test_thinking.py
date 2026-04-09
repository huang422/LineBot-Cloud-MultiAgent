"""Manual smoke test for reasoning/thinking verification.

Usage:
    python -m tests.test_thinking

This file intentionally avoids pytest-style ``test_*`` functions so it can live
under ``tests/`` without being collected during the normal automated suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from dotenv import load_dotenv


async def run_openrouter_reasoning():
    """Test OpenRouter reasoning with a known reasoning-capable model."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("[SKIP] OpenRouter: no valid API key")
        return

    import httpx

    model = os.getenv("ORCHESTRATOR_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    print(f"\n{'='*60}")
    print(f"[OpenRouter] Testing reasoning with {model}")
    print(f"{'='*60}")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 17 * 23? Think step by step."}
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "reasoning": {
            "enabled": True,
            "effort": "high",
            "exclude": False,
        },
        "include_reasoning": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/linebot-cloud-agent",
        "X-OpenRouter-Title": "LineBot-CloudAgent",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    print(f"HTTP Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    usage = data.get("usage", {})

    reasoning = msg.get("reasoning")
    reasoning_content = msg.get("reasoning_content")
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    if not reasoning_tokens:
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)

    print(f"  reasoning field  : {_preview(reasoning, 100)}")
    print(f"  reasoning_tokens : {reasoning_tokens}")

    has = bool(reasoning or reasoning_content or reasoning_tokens > 0)
    print(f"  {'✅' if has else '❌'} OpenRouter reasoning: {'CONFIRMED' if has else 'NOT DETECTED'}")


async def run_nvidia_thinking():
    """Test NVIDIA Gemma 4 thinking with fixed top-level chat_template_kwargs."""
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("[SKIP] NVIDIA: no valid API key")
        return

    import httpx

    model = os.getenv("NVIDIA_THINKING_MODEL", "google/gemma-4-31b-it")
    print(f"\n{'='*60}")
    print(f"[NVIDIA] Testing thinking with {model}")
    print(f"  (fixed: chat_template_kwargs as top-level param)")
    print(f"{'='*60}")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 17 * 23? Think step by step."}
        ],
        "max_tokens": 4096,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    print(f"HTTP Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning")
    reasoning_content = msg.get("reasoning_content")
    usage = data.get("usage", {})
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    if not reasoning_tokens:
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)

    has_think_tags = "<think>" in content
    if has_think_tags:
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            print(f"  <think> tags     : YES ({len(think_match.group(1).strip())} chars)")
    else:
        print(f"  <think> tags     : NO")

    print(f"  reasoning field  : {_preview(reasoning, 100)}")
    print(f"  reasoning_content: {_preview(reasoning_content, 100)}")
    print(f"  reasoning_tokens : {reasoning_tokens}")
    print(f"  answer preview   : {_preview(content, 100)}")

    has = bool(reasoning or reasoning_content or has_think_tags or reasoning_tokens > 0)
    print(f"  {'✅' if has else '❌'} NVIDIA thinking: {'CONFIRMED' if has else 'NOT DETECTED'}")


async def run_nvidia_provider_class():
    """Test through the actual NvidiaProvider class (end-to-end with fix)."""
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("[SKIP] NVIDIA provider class: no valid API key")
        return

    print(f"\n{'='*60}")
    print(f"[NVIDIA Provider Class] End-to-end test")
    print(f"{'='*60}")

    from src.utils.rate_tracker import RateTracker
    from src.providers.nvidia_provider import NvidiaProvider

    model = os.getenv("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b")
    provider = NvidiaProvider(
        api_key,
        RateTracker(),
        thinking_enabled=True,
        thinking_budget=4096,
        thinking_model=os.getenv("NVIDIA_THINKING_MODEL", "google/gemma-4-31b-it"),
        primary_model=os.getenv("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b"),
    )

    try:
        resp = await provider.generate(
            model=model,
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
            temperature=0.6,
            max_tokens=2048,
            require_reasoning_tokens=True,
        )

        print(f"  text             : {_preview(resp.text, 100)}")
        print(f"  reasoning_content: {_preview(resp.reasoning_content, 100)}")
        print(f"  model            : {resp.model}")

        has = bool(resp.reasoning_content)
        print(f"  {'✅' if has else '❌'} NvidiaProvider thinking: {'CONFIRMED' if has else 'NOT DETECTED'}")
    finally:
        await provider.close()


def _preview(val, max_len=150):
    if val is None:
        return "None"
    if not isinstance(val, str):
        val = str(val)
    if not val.strip():
        return "(empty)"
    return val[:max_len] + ("..." if len(val) > max_len else "")


async def main():
    load_dotenv()

    print("=" * 60)
    print("Thinking/Reasoning Smoke Test (post-fix)")
    print("=" * 60)

    await run_openrouter_reasoning()
    await run_nvidia_thinking()
    await run_nvidia_provider_class()

    print(f"\n{'='*60}")
    print("All tests done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
