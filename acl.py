"""Slack channel-membership lookup with a short-lived cache.

`channels_for_user` returns None when membership cannot be established. None
means "narrow the scope", never "allow everything" — callers must treat it as
a failure, not as an empty allow-list.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TTL_SECONDS = 300.0
_SLACK_TIMEOUT = 5
_PAGE_LIMIT = 200
_MAX_PAGES = 20  # same safety cap gateway/channel_directory.py uses

# Only successful lookups are cached. Caching a failure would pin a user into
# narrow scope for the whole TTL after Slack recovers.
_CACHE: Dict[str, Tuple[float, FrozenSet[str]]] = {}
_LOCK = threading.Lock()

# Captured from the pre_gateway_dispatch hook. The plugin has no direct handle
# on the running GatewayRunner, and this is the documented way to get one.
_GATEWAY = None


def capture_gateway(event=None, gateway=None, session_store=None, **_kwargs):
    """Observer hook. Records the gateway and never influences dispatch.

    Returning None unconditionally means the hook's fail-open semantics do not
    matter: if it never fires, _GATEWAY stays None and lookups fail closed.
    """
    global _GATEWAY
    if gateway is not None:
        _GATEWAY = gateway
    return None


def _bot_tokens() -> List[str]:
    """Slack bot tokens, preferring the live adapters over configuration.

    The adapter token belongs to the workspace that actually received the
    message, which is what we want under multiplex_profiles.
    """
    tokens: List[str] = []
    gw = _GATEWAY
    if gw is not None:
        adapter_maps = [getattr(gw, "adapters", {}) or {}]
        adapter_maps += list((getattr(gw, "_profile_adapters", {}) or {}).values())
        for mapping in adapter_maps:
            try:
                adapters = mapping.values()
            except Exception:
                continue
            for adapter in adapters:
                for client in (getattr(adapter, "_team_clients", {}) or {}).values():
                    token = getattr(client, "token", None)
                    if token:
                        tokens.append(token)

    if tokens:
        return tokens

    # Fallback. get_secret raises outside a secret scope, which under
    # multiplex_profiles is the normal case — swallow into "no token".
    try:
        from agent.secret_scope import get_secret

        raw = get_secret("SLACK_BOT_TOKEN") or ""
        return [part.strip() for part in raw.split(",") if part.strip()]
    except Exception:
        return []


def bot_tokens() -> List[str]:
    """Public alias for sibling modules that need to build their own client."""
    return _bot_tokens()


def _fetch_for_token(token: str, user_id: str) -> Optional[FrozenSet[str]]:
    """Channels `user_id` belongs to, as visible to this token. None on error."""
    try:
        from slack_sdk import WebClient
    except Exception:
        logger.error("acl: slack_sdk unavailable")
        return None

    client = WebClient(token=token, timeout=_SLACK_TIMEOUT)
    found: set = set()
    cursor = ""
    for _ in range(_MAX_PAGES):
        try:
            resp = client.users_conversations(
                user=user_id,
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=_PAGE_LIMIT,
                cursor=cursor or None,
            )
        except Exception as exc:
            logger.warning("acl: users.conversations failed: %s", exc)
            return None

        if not resp.get("ok"):
            logger.warning("acl: users.conversations not ok: %s", resp.get("error"))
            return None

        for channel in resp.get("channels") or []:
            cid = channel.get("id")
            if cid:
                found.add(cid)

        cursor = ((resp.get("response_metadata") or {}).get("next_cursor") or "").strip()
        if not cursor:
            break

    return frozenset(found)


def channels_for_user(user_id: str) -> Optional[FrozenSet[str]]:
    """Channel ids the user belongs to, or None if it cannot be determined."""
    if not user_id:
        return None

    now = time.monotonic()
    with _LOCK:
        cached = _CACHE.get(user_id)
        if cached and (now - cached[0]) < _TTL_SECONDS:
            return cached[1]

    tokens = _bot_tokens()
    if not tokens:
        logger.warning("acl: no Slack token available, cannot resolve membership")
        return None

    union: set = set()
    any_success = False
    for token in tokens:
        result = _fetch_for_token(token, user_id)
        if result is not None:
            any_success = True
            union |= result

    if not any_success:
        return None

    resolved = frozenset(union)
    with _LOCK:
        _CACHE[user_id] = (time.monotonic(), resolved)
    return resolved


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
