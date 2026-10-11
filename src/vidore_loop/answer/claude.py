"""One sealed Claude call through the Agent SDK, billed to the subscription (M4).

After tau2-loop's `llm/core.py`: the session can only answer. It has no built-in
tools, no MCP servers, no settings files, a working directory outside any
checkout, and an environment reduced to an allow-list (the SDK merges
`options.env` over the parent's, so every other name is passed blank). Under
`BILLING=subscription` an `ANTHROPIC_API_KEY` is refused outright, because with a
key present the CLI bills per token.

The prompt goes in as content blocks, so page images travel as base64 image
blocks beside their OCR text. A subscription window that has run out is waited
for, not failed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5-5",
}
EFFORTS = ("low", "medium", "high", "xhigh", "max")

RETRIES = 3
RETRY_WAIT_S = (2.0, 6.0, 15.0)
MAX_WAIT_S = 6 * 3600

# What the CLI child may inherit: what runs it, its login, and proxy / certificate settings.
ENV_ALLOW = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "TERM",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    }
)
ENV_FIXED = {
    "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
    # stops the CLI naming every session with a separate Haiku call
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDECODE": "",
}

Block = dict[str, Any]


class BillingError(RuntimeError):
    """The environment would bill the wrong way, or cannot bill at all."""


class ClaudeError(RuntimeError):
    """The SDK gave no reply after its retries."""


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cost_usd: float | None = None  # what the call would cost per token; the subscription pays
    ms: int = 0
    model: str = ""
    structured: Any = None
    error: str | None = None

    def usage(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        d.pop("structured")
        return d


def resolve(name: str) -> str:
    """`sonnet` → `claude-sonnet-5`; a full model id passes through."""
    return MODELS.get(name.strip().lower(), name.strip())


def billing() -> str:
    return os.environ.get("BILLING", "subscription").strip().lower()


def check_billing() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if billing() == "subscription" and key:
        raise BillingError(
            "ANTHROPIC_API_KEY is set while BILLING=subscription: unset it, or the CLI bills per token"
        )
    if billing() == "api" and not key:
        raise BillingError("BILLING=api but ANTHROPIC_API_KEY is not set")


def available() -> tuple[bool, str]:
    """Whether a call could go out now, and why not when it cannot (for the viewer)."""
    try:
        check_billing()
        import claude_agent_sdk  # noqa: F401
    except (BillingError, ImportError) as e:
        return False, str(e)
    return True, billing()


def sealed_env(parent: dict[str, str] | None = None) -> dict[str, str]:
    parent = dict(os.environ if parent is None else parent)
    env = {k: (v if k in ENV_ALLOW or k.startswith("LC_") else "") for k, v in parent.items()}
    env.update(ENV_FIXED)
    env["ANTHROPIC_API_KEY"] = parent.get("ANTHROPIC_API_KEY", "") if billing() == "api" else ""
    return env


def session_dir() -> Path:
    p = Path(tempfile.gettempdir()) / "vidoreloop-sdk"
    p.mkdir(parents=True, exist_ok=True)
    return p


def text_block(text: str) -> Block:
    return {"type": "text", "text": text}


def image_block(path: Path) -> Block:
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


async def _message(blocks: list[Block]) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "user",
        "message": {"role": "user", "content": blocks},
        "parent_tool_use_id": None,
        "session_id": "default",
    }


async def _query(
    system: str, blocks: list[Block], model: str, effort: str, schema: dict[str, Any] | None
) -> Reply:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        tools=[],  # no built-in tools: the model can only answer
        allowed_tools=[],
        mcp_servers={},
        strict_mcp_config=True,
        permission_mode="bypassPermissions",
        max_turns=4,  # one reply; headroom because the CLI has ended a tool-less reply as 'max turns'
        cwd=str(session_dir()),
        env=sealed_env(),
        setting_sources=[],
        effort=effort,
        output_format={"type": "json_schema", "schema": schema} if schema else None,
    )
    texts: list[str] = []
    rep = Reply("", model=model)
    t0 = time.time()
    async for msg in query(prompt=_message(blocks), options=options):
        if isinstance(msg, AssistantMessage):
            texts += [b.text for b in msg.content if isinstance(b, TextBlock)]
        elif isinstance(msg, ResultMessage):
            u = msg.usage or {}
            rep.input_tokens = (
                int(u.get("input_tokens", 0))
                + int(u.get("cache_read_input_tokens", 0))
                + int(u.get("cache_creation_input_tokens", 0))
            )
            rep.output_tokens = int(u.get("output_tokens", 0))
            rep.cache_read = int(u.get("cache_read_input_tokens", 0))
            rep.cost_usd = msg.total_cost_usd
            if msg.is_error:
                rep.error = f"{msg.subtype}: {(msg.errors or [''])[0]}"[:500]
            rep.structured = getattr(msg, "structured_output", None)
            if not texts and rep.structured is not None:
                texts.append(json.dumps(rep.structured))
            elif not texts and msg.result:
                texts.append(str(msg.result))
    rep.text = "\n".join(t for t in texts if t).strip()
    rep.ms = int((time.time() - t0) * 1000)
    return rep


_RESET_RE = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)", re.I)


def seconds_until_reset(error: str, now: datetime | None = None) -> int | None:
    """The wait a 'session limit · resets 4:40pm' error asks for; None for any other error."""
    if "limit" not in error.lower() or "reset" not in error.lower():
        return None
    m = _RESET_RE.search(error)
    now = now or datetime.now()
    if not m:
        return 15 * 60
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    hour = hour % 12 + (12 if ampm == "pm" else 0)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return min(int((target - now).total_seconds()) + 60, MAX_WAIT_S)


def ask(
    system: str,
    blocks: list[Block],
    model: str = "sonnet",
    effort: str = "medium",
    schema: dict[str, Any] | None = None,
) -> Reply:
    """One call on a private event loop (safe from worker threads); raises ClaudeError on no reply."""
    check_billing()
    model = resolve(model)
    last = Reply("", model=model)
    attempt = 0
    while attempt < RETRIES:
        loop = asyncio.new_event_loop()
        try:
            last = loop.run_until_complete(_query(system, blocks, model, effort, schema))
        except Exception as e:  # noqa: BLE001 - transport errors are retried like HTTP ones
            last = Reply("", model=model, error=f"{type(e).__name__}: {e}"[:500])
        finally:
            loop.close()
        if last.text:
            return last
        wait = seconds_until_reset(last.error or "")
        if wait is not None:
            print(
                f"[claude] subscription limit; waiting {wait // 60} min for the reset", flush=True
            )
            time.sleep(wait)
            continue  # the wait is not an attempt
        attempt += 1
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT_S[attempt - 1])
    raise ClaudeError(f"no reply from {model}: {last.error or 'empty reply'}")
