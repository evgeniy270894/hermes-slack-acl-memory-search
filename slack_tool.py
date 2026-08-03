"""Read and search Slack, scoped to what the asking user may already see.

Three actions:

  read_channel(channel)             conversations.history
  read_thread(channel, thread_ts)   conversations.replies
  search(query)                     assistant.search.context (Real-time Search)

Scope rule, decided by who will READ the answer — the same rule the scoped
session_search uses:

  asking in a channel  -> that channel only. Whatever the bot says there is
                          read by everyone present, so nothing else may be
                          pulled in.
  asking in a DM       -> the DM itself plus channels the asker belongs to.
                          They could open those channels themselves anyway.
  anything else        -> refuse.

`search` is the one exception, and deliberately: with a bot token the
Real-time Search API only ever reaches PUBLIC channels (private/DM search
needs user-token scopes we do not hold), and every workspace member can read
any public channel by joining it. So public results are not a disclosure in
either context. Each hit is labelled with its channel so the answer stays
attributable.

Never trust the "[Slack app context: user is viewing channel C...]" line the
adapter prepends to the message: it is injected into the message TEXT, so a
user can type the identical string and name any channel. It is a convenience
default only — every channel argument is checked against real membership.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import acl, action_token, scope

logger = logging.getLogger(__name__)

TOOL_NAME = "slack"
# Must be exactly "slack": gateway/session.py:_slack_tools_loaded() flips the
# agent's Slack capability note from "you cannot search channel history" to
# "use your Slack tools" only when a toolset with THIS name is enabled for the
# slack platform. Any other name leaves the model believing it has no tools.
TOOLSET = "slack"

_SLACK_TIMEOUT = 15
_DEFAULT_LIMIT = 40
_MAX_LIMIT = 150
_MAX_TEXT_CHARS = 1200
_SEARCH_MAX = 20  # Slack hard cap for assistant.search.context

_REFUSAL = "Not available here: this conversation gives no access to that Slack content."

_USER_NAMES: Dict[str, str] = {}
_USER_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def _client():
    """A Slack WebClient built from a live adapter token, or None."""
    tokens = acl.bot_tokens()
    if not tokens:
        return None
    try:
        from slack_sdk import WebClient
    except Exception:
        logger.error("slack tool: slack_sdk unavailable")
        return None
    return WebClient(token=tokens[0], timeout=_SLACK_TIMEOUT)


def _call(client, method: str, **kwargs):
    """One Slack call with Retry-After-aware backoff.

    The adapter's own retry ignores Retry-After and computes 1s/2s from the
    attempt counter; that is fine at Tier 3 and wrong everywhere else, so it is
    not copied here.
    """
    for attempt in range(3):
        try:
            return getattr(client, method)(**kwargs)
        except Exception as exc:
            retry_after = getattr(getattr(exc, "response", None), "headers", {}) or {}
            wait = retry_after.get("Retry-After") or retry_after.get("retry-after")
            if wait is None or attempt == 2:
                raise
            time.sleep(min(float(wait), 30.0))
    raise RuntimeError("unreachable")


def _user_name(client, user_id: str) -> str:
    if not user_id:
        return ""
    with _USER_LOCK:
        cached = _USER_NAMES.get(user_id)
    if cached:
        return cached
    name = user_id
    try:
        resp = _call(client, "users_info", user=user_id)
        profile = (resp.get("user") or {}).get("profile") or {}
        name = (
            profile.get("display_name")
            or profile.get("real_name")
            or (resp.get("user") or {}).get("name")
            or user_id
        )
    except Exception:
        pass
    with _USER_LOCK:
        _USER_NAMES[user_id] = name
    return name


def _render(client, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for msg in messages:
        text = (msg.get("text") or "").strip()
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS] + " […truncated]"
        out.append(
            {
                "ts": msg.get("ts"),
                "author": _user_name(client, msg.get("user") or "")
                or (msg.get("username") or "bot"),
                "text": text,
                "thread_ts": msg.get("thread_ts"),
                "reply_count": msg.get("reply_count", 0),
            }
        )
    return out


def _clamp(value: Any, default: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, maximum))


# --------------------------------------------------------------------------- #
# scope
# --------------------------------------------------------------------------- #

def _resolve() -> Tuple[Optional[scope.AskingContext], Optional[scope.Scope]]:
    """Caller identity and permitted channel set. (None, None) means refuse."""
    ctx = scope.resolve_asking_context(None, "")
    if ctx is None:
        return None, None
    sc = scope.resolve_scope(ctx, acl)
    if sc is None:
        return None, None
    return ctx, sc


def _target_channel(requested: Any, ctx: scope.AskingContext, sc: scope.Scope) -> Optional[str]:
    """Validate the requested channel against the permitted set.

    Defaults to the current chat when the model omits it, which is the common
    case for "what was said here".
    """
    channel = (str(requested).strip() if requested else "") or ctx.chat_id
    if channel not in sc.chat_ids:
        logger.info(
            "slack tool: refusing channel %r for user %s (scope=%s)",
            channel, ctx.user_id, sc.reason,
        )
        return None
    return channel


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #

def _read_channel(client, args: dict, ctx, sc) -> dict:
    channel = _target_channel(args.get("channel"), ctx, sc)
    if channel is None:
        return {"error": _REFUSAL}

    limit = _clamp(args.get("limit"), _DEFAULT_LIMIT, _MAX_LIMIT)
    kwargs = {"channel": channel, "limit": limit}
    if args.get("oldest"):
        kwargs["oldest"] = str(args["oldest"])

    resp = _call(client, "conversations_history", **kwargs)
    if not resp.get("ok"):
        return {"error": f"slack: {resp.get('error')}"}

    messages = resp.get("messages") or []
    return {
        "channel": channel,
        "count": len(messages),
        "has_more": bool(resp.get("has_more")),
        # conversations.history returns only top-level messages; replies live
        # behind conversations.replies, so surface where they are.
        "note": "Thread replies are not included — use read_thread with a ts that has reply_count > 0.",
        "messages": _render(client, messages),
    }


def _read_thread(client, args: dict, ctx, sc) -> dict:
    channel = _target_channel(args.get("channel"), ctx, sc)
    if channel is None:
        return {"error": _REFUSAL}

    thread_ts = str(args.get("thread_ts") or "").strip()
    if not thread_ts:
        return {"error": "read_thread needs thread_ts (the ts of the thread's parent message)."}

    limit = _clamp(args.get("limit"), _DEFAULT_LIMIT, _MAX_LIMIT)
    resp = _call(client, "conversations_replies", channel=channel, ts=thread_ts, limit=limit)
    if not resp.get("ok"):
        return {"error": f"slack: {resp.get('error')}"}

    messages = resp.get("messages") or []
    return {
        "channel": channel,
        "thread_ts": thread_ts,
        "count": len(messages),
        "messages": _render(client, messages),
    }


def _search(client, args: dict, ctx, sc) -> dict:
    query = str(args.get("query") or "").strip()
    if not query:
        return {"error": "search needs a query."}

    token = action_token.get(ctx.chat_id, ctx.user_id)
    if not token:
        return {
            "error": (
                "Slack search is unavailable for this turn: no action_token was "
                "captured. Slack only issues one on a fresh user message, so ask "
                "the user to send the request again."
            )
        }

    limit = _clamp(args.get("limit"), _SEARCH_MAX, _SEARCH_MAX)
    resp = _call(
        client,
        "assistant_search_context",
        query=query,
        limit=limit,
        action_token=token,
    )
    if not resp.get("ok"):
        return {"error": f"slack: {resp.get('error')}"}

    results = ((resp.get("results") or {}).get("messages")) or []
    hits = []
    for item in results:
        text = (item.get("content") or item.get("text") or "").strip()
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS] + " […truncated]"
        hits.append(
            {
                "channel_id": (item.get("channel") or {}).get("id") if isinstance(item.get("channel"), dict) else item.get("channel_id"),
                "channel_name": (item.get("channel") or {}).get("name") if isinstance(item.get("channel"), dict) else None,
                "author": item.get("author_name") or item.get("author_user_id"),
                "ts": item.get("message_ts") or item.get("ts"),
                "permalink": item.get("permalink"),
                "text": text,
            }
        )
    return {
        "query": query,
        "count": len(hits),
        "scope": "public channels only (bot-token Real-time Search)",
        "results": hits,
    }


_ACTIONS = {
    "read_channel": _read_channel,
    "read_thread": _read_thread,
    "search": _search,
}


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #

def check() -> Tuple[bool, str]:
    """Registry check_fn — hide the tool when it cannot possibly work."""
    if not acl.bot_tokens():
        return False, "no Slack bot token available"
    return True, ""


def handler(args: dict, **_kwargs) -> str:
    try:
        action = str((args or {}).get("action") or "").strip()
        fn = _ACTIONS.get(action)
        if fn is None:
            return json.dumps(
                {"error": f"unknown action {action!r}; expected one of {sorted(_ACTIONS)}"},
                ensure_ascii=False,
            )

        ctx, sc = _resolve()
        if ctx is None or sc is None:
            return json.dumps({"error": _REFUSAL}, ensure_ascii=False)

        client = _client()
        if client is None:
            return json.dumps({"error": "Slack client unavailable."}, ensure_ascii=False)

        return json.dumps(fn(client, args or {}, ctx, sc), ensure_ascii=False)
    except Exception as exc:
        # Deny on any unexpected failure rather than leaking a partial result.
        logger.warning("slack tool failed: %s", exc, exc_info=True)
        return json.dumps({"error": "Slack lookup failed and was refused."}, ensure_ascii=False)


def build_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Read or search Slack. Actions: read_channel (recent messages of a "
                "channel; defaults to the current one), read_thread (replies of one "
                "thread), search (semantic search across public channels). Only "
                "content the person you are replying to may already see is returned; "
                "out-of-scope requests are refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(_ACTIONS),
                        "description": "Which operation to perform.",
                    },
                    "channel": {
                        "type": "string",
                        "description": (
                            "Channel id (C…/G…/D…). Omit to use the current conversation. "
                            "read_channel and read_thread only."
                        ),
                    },
                    "thread_ts": {
                        "type": "string",
                        "description": "Parent message ts of the thread. read_thread only.",
                    },
                    "oldest": {
                        "type": "string",
                        "description": "Unix ts; only messages after it. read_channel only.",
                    },
                    "query": {
                        "type": "string",
                        "description": "What to look for. search only.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max results (default {_DEFAULT_LIMIT}, search caps at {_SEARCH_MAX}).",
                    },
                },
                "required": ["action"],
            },
        },
    }


def mark_untrusted() -> None:
    """Make fetched Slack text arrive as data, not instructions.

    `_maybe_wrap_untrusted` classifies by tool NAME against a module-level
    frozenset plus the browser_/mcp_ prefixes, so a plugin tool gets no wrapper
    unless the set is extended. The predicate reads the global on every call,
    so rebinding it here takes effect immediately.
    """
    try:
        import agent.tool_dispatch_helpers as helpers

        if TOOL_NAME not in helpers._UNTRUSTED_TOOL_NAMES:
            helpers._UNTRUSTED_TOOL_NAMES = frozenset(
                helpers._UNTRUSTED_TOOL_NAMES | {TOOL_NAME}
            )
            logger.info("slack tool: output marked as untrusted content")
    except Exception as exc:
        # Worth shouting about: without this, channel text reaches the model as
        # trusted instructions, which is the whole indirect-injection surface.
        logger.error("slack tool: could not mark output untrusted: %s", exc)
