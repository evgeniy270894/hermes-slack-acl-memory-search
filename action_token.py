"""Capture Slack's per-interaction ``action_token`` off the inbound event.

``assistant.search.context`` called with a *bot* token requires an
``action_token``; Slack delivers it inside the triggering ``message.*`` /
``app_mention`` payload and documents it only as "short-lived", with no
published TTL. Hermes has no plumbing for it (no occurrence anywhere in the
tree at v2026.7.30), but the Slack adapter does preserve the raw event as
``MessageEvent.raw_message``, and ``pre_gateway_dispatch`` hands us that
event — so this is the seam.

The token authorises a search *on behalf of the user who triggered it*, so it
is cached per (chat_id, user_id) and never shared across callers.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Slack publishes no TTL. Keep it well under any plausible expiry: a stale
# token yields an API error, and re-prompting is better than a long window of
# a user-scoped credential sitting in memory.
_TTL_SECONDS = 120.0
_MAX_ENTRIES = 500

_CACHE: Dict[Tuple[str, str], Tuple[float, str]] = {}
_LOCK = threading.Lock()

# One-shot: confirms empirically whether Slack ships the field at all, without
# ever writing the value to a log.
_SHAPE_LOGGED = False


def _prune(now: float) -> None:
    stale = [k for k, (ts, _) in _CACHE.items() if now - ts > _TTL_SECONDS]
    for key in stale:
        del _CACHE[key]
    if len(_CACHE) > _MAX_ENTRIES:
        for key in sorted(_CACHE, key=lambda k: _CACHE[k][0])[: len(_CACHE) - _MAX_ENTRIES]:
            del _CACHE[key]


def capture(event: Any = None, **_kwargs) -> None:
    """pre_gateway_dispatch observer. Always returns None — never influences dispatch."""
    global _SHAPE_LOGGED
    try:
        raw = getattr(event, "raw_message", None)
        if not isinstance(raw, dict):
            return None

        if not _SHAPE_LOGGED:
            _SHAPE_LOGGED = True
            logger.info(
                "action_token probe: inbound slack event carries fields %s (action_token present: %s)",
                sorted(raw.keys()),
                "action_token" in raw,
            )

        token = raw.get("action_token")
        if not isinstance(token, str) or not token:
            return None

        chat_id = str(raw.get("channel") or raw.get("channel_id") or "")
        user_id = str(raw.get("user") or raw.get("user_id") or "")
        if not chat_id or not user_id:
            return None

        now = time.monotonic()
        with _LOCK:
            _CACHE[(chat_id, user_id)] = (now, token)
            _prune(now)
    except Exception as exc:  # never break dispatch over telemetry
        logger.debug("action_token capture failed: %s", exc)
    return None


def get(chat_id: str, user_id: str) -> Optional[str]:
    """The freshest token for this caller, or None when absent/expired."""
    if not chat_id or not user_id:
        return None
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get((chat_id, user_id))
        if entry is None:
            return None
        captured_at, token = entry
        if now - captured_at > _TTL_SECONDS:
            del _CACHE[(chat_id, user_id)]
            return None
        return token


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
