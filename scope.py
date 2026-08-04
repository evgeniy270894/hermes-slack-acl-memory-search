"""Resolve who is asking and what they may see.

Pure logic plus two cheap reads (session env vars, one sessions row). No Slack
calls here — membership lookup lives in acl.py so this module stays testable
without network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)

# Slack channel-id prefixes. Used only as a fallback when the sessions row is
# not yet written (first turn of a brand-new session).
_DM_PREFIX = "D"
_GROUP_PREFIXES = ("C", "G")


@dataclass(frozen=True)
class AskingContext:
    platform: str
    chat_id: str
    user_id: str
    chat_type: str  # "dm" | "group"
    session_id: str


@dataclass(frozen=True)
class Scope:
    """What the asker is allowed to reach.

    A candidate session row is in scope iff its chat_id is in `chat_ids`, OR
    its user_id is in `user_ids` AND it is a DM. The dm conjunct is essential:
    without it, the asker appearing as user_id on a channel row would admit
    every message in that channel.
    """

    chat_ids: FrozenSet[str]
    user_ids: FrozenSet[str]
    reason: str


def _session_env(name: str) -> str:
    try:
        from gateway.session_context import get_session_env

        return (get_session_env(name, "") or "").strip()
    except Exception:
        return ""


def _chat_type_from_prefix(chat_id: str) -> Optional[str]:
    if not chat_id:
        return None
    if chat_id.startswith(_DM_PREFIX):
        return "dm"
    if chat_id.startswith(_GROUP_PREFIXES):
        return "group"
    return None


def resolve_asking_context(db, current_session_id: str) -> Optional[AskingContext]:
    """Identify the caller, or None when it cannot be established.

    None always means refuse — never fall back to an unscoped search.
    """
    platform = _session_env("HERMES_SESSION_PLATFORM")
    chat_id = _session_env("HERMES_SESSION_CHAT_ID")
    user_id = _session_env("HERMES_SESSION_USER_ID")

    if platform != "slack" or not chat_id or not user_id:
        logger.info(
            "scope: refusing, incomplete identity (platform=%r chat_id=%r user_id=%r)",
            platform,
            bool(chat_id),
            bool(user_id),
        )
        return None

    chat_type = None
    row = None
    if current_session_id:
        try:
            row = db.get_session(current_session_id)
        except Exception:
            logger.warning("scope: get_session failed for current session", exc_info=True)
            return None

    if row:
        row_chat_id = (row.get("chat_id") or "").strip()
        # A mismatch means the env vars and the stored row disagree about which
        # conversation this is. Never guess which one is authoritative.
        if row_chat_id and row_chat_id != chat_id:
            logger.error("scope: refusing, chat_id mismatch between env and session row")
            return None
        chat_type = (row.get("chat_type") or "").strip() or None

    if not chat_type:
        # Session row not written yet (first turn) — derive from the Slack id.
        chat_type = _chat_type_from_prefix(chat_id)

    # A Slack group DM (mpim) arrives as chat_type "dm", but it is a room other
    # people are reading — and "dm" earns the wide scope (the asker's own
    # sessions plus every channel they belong to). Trust the id shape over the
    # row: only a real 1:1 DM starts with "D". Without this, one member asking a
    # normal question prints another member's private conversation into the room.
    if chat_type == "dm" and not chat_id.startswith(_DM_PREFIX):
        logger.info("scope: %r is not a 1:1 DM, narrowing verdict to group", chat_id)
        chat_type = "group"

    if chat_type not in ("dm", "group"):
        logger.info("scope: refusing, unresolved chat_type for chat_id=%r", chat_id)
        return None

    return AskingContext(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        chat_type=chat_type,
        session_id=current_session_id or "",
    )


def resolve_scope(ctx: AskingContext, acl) -> Optional[Scope]:
    """Turn an identity into a permitted set.

    Channel: only that channel — whatever the bot answers there is read by
    everyone present, so nothing private may be pulled in.

    DM: the asker's own sessions plus channels they belong to — the answer is
    seen only by them, and they could read those channels themselves anyway.
    """
    if ctx.chat_type == "group":
        return Scope(chat_ids=frozenset({ctx.chat_id}), user_ids=frozenset(), reason="channel")

    if ctx.chat_type == "dm":
        member_channels = acl.channels_for_user(ctx.user_id)
        if member_channels is None:
            # Slack unreachable / no token / timeout. Narrow, never widen.
            logger.warning("scope: ACL unavailable for %s, narrowing to current chat", ctx.user_id)
            return Scope(
                chat_ids=frozenset({ctx.chat_id}),
                user_ids=frozenset(),
                reason="dm-acl-unavailable",
            )
        return Scope(
            chat_ids=frozenset({ctx.chat_id}) | member_channels,
            user_ids=frozenset({ctx.user_id}),
            reason="dm",
        )

    return None
