"""The scoped replacement for session_search.

Does not reimplement search. Resolves the caller's scope, then delegates to the
original function with a filtering DB proxy substituted for `db`.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Optional

from . import acl
from .db_proxy import ScopedSessionDB
from .scope import resolve_asking_context, resolve_scope

logger = logging.getLogger(__name__)

# Set by __init__._install_module_patch before the wrapper can be called.
_ORIGINAL = None

_REFUSAL = "session_search is unavailable in this context (identity could not be resolved)"


def set_original(func) -> None:
    global _ORIGINAL
    _ORIGINAL = func


def _tool_error(message: str, **extra) -> str:
    from tools.registry import tool_error

    return tool_error(message, **extra)


def _not_found(session_id: str) -> str:
    # Deliberately identical to the built-in's miss text so an out-of-scope id
    # is indistinguishable from a nonexistent one — no existence oracle.
    return _tool_error(f"session_id not found: {session_id}", success=False)


def _strip_profile_prefix(session_id: Optional[str]) -> Optional[str]:
    """Drop an embedded "profile/id" prefix without adopting the profile."""
    if isinstance(session_id, str) and "/" in session_id:
        return session_id.split("/", 1)[1]
    return session_id


def _collect_session_ids(payload: Any, out: set) -> None:
    if isinstance(payload, dict):
        for key in ("session_id", "parent_session_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                out.add(value)
        for value in payload.values():
            _collect_session_ids(value, out)
    elif isinstance(payload, list):
        for item in payload:
            _collect_session_ids(item, out)


def scoped_session_search(
    query: str = "",
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
    session_id: str = None,
    around_message_id: int = None,
    window: int = 5,
    sort: str = None,
    profile: str = None,
) -> str:
    try:
        if _ORIGINAL is None:
            return _tool_error("session_search is not properly initialised", success=False)

        # Cross-profile reads open other profiles' databases directly, bypassing
        # the proxy entirely. Refuse both spellings.
        if profile:
            logger.warning("scoped search: refused cross-profile read (profile=%r)", profile)
            return _tool_error("cross-profile session reads are disabled by policy", success=False)
        session_id = _strip_profile_prefix(session_id)

        if db is None:
            from hermes_state import SessionDB

            db = SessionDB()

        ctx = resolve_asking_context(db, current_session_id)
        if ctx is None:
            return _tool_error(_REFUSAL, success=False)

        scope = resolve_scope(ctx, acl)
        if scope is None:
            return _tool_error(_REFUSAL, success=False)

        logger.info(
            "scoped search: user=%s chat=%s type=%s reason=%s chats=%d",
            ctx.user_id,
            ctx.chat_id,
            ctx.chat_type,
            scope.reason,
            len(scope.chat_ids),
        )

        proxy = ScopedSessionDB(db, scope)

        # Validate before delegating: on a read miss the original falls back to
        # _locate_session_db, which scans every profile's database and never
        # sees our proxy.
        if session_id and not proxy.lineage_allowed(session_id):
            logger.info("scoped search: denied out-of-scope session_id")
            return _not_found(session_id)

        result = _ORIGINAL(
            query=query,
            role_filter=role_filter,
            limit=limit,
            db=proxy,
            current_session_id=current_session_id,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            sort=sort,
            profile=None,
        )

        # Second, independent gate: whatever came back must still be in scope.
        # Catches any future path that reaches around the proxy.
        try:
            payload = json.loads(result)
        except Exception:
            return result

        if isinstance(payload, dict) and payload.get("profile"):
            logger.critical("scoped search: result carried a foreign profile marker")
            return _tool_error("internal scope violation", success=False)

        seen: set = set()
        _collect_session_ids(payload, seen)
        for sid in seen:
            if not proxy.lineage_allowed(sid):
                logger.critical("scoped search: result contained out-of-scope session %s", sid)
                return _tool_error("internal scope violation", success=False)

        return result

    except Exception:
        # Fail closed. Any unexpected error must not degrade into an
        # unfiltered search.
        logger.critical("scoped search: refusing after unexpected error", exc_info=True)
        return _tool_error("session_search failed and was refused for safety", success=False)


scoped_session_search.__hermes_acl_wrapped__ = True


def build_schema() -> dict:
    """Built-in schema minus the cross-profile parameter."""
    from tools.session_search_tool import SESSION_SEARCH_SCHEMA

    schema = copy.deepcopy(SESSION_SEARCH_SCHEMA)
    params = schema.get("parameters") or {}
    props = params.get("properties") or {}
    props.pop("profile", None)

    description = schema.get("description") or ""
    schema["description"] = (
        description
        + "\n\nResults are scoped to conversations you are entitled to see: in a "
        "channel, only that channel; in a direct message, your own conversations "
        "plus channels you belong to."
    )
    return schema


def check() -> bool:
    from tools.session_search_tool import check_session_search_requirements

    return check_session_search_requirements()


def handler(args: dict, **kwargs) -> str:
    args = args or {}
    return scoped_session_search(
        query=args.get("query", ""),
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=kwargs.get("db"),
        current_session_id=kwargs.get("current_session_id"),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        sort=args.get("sort"),
        profile=args.get("profile"),
    )
