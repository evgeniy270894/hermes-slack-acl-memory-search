"""A SessionDB wrapper that only reveals rows inside the caller's scope.

Explicit allow-list, not attribute forwarding: an unknown method must raise
loudly rather than pass through unfiltered. A Hermes upgrade that adds a new
`db.` call inside session_search should break visibly, not widen silently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .scope import Scope

logger = logging.getLogger(__name__)

# Rows scanned before filtering. The built-in caps discovery at 300 rows, all
# of which could be out of scope in a busy workspace — widen, filter, then cut
# back to the caller's limit.
_WIDEN_FACTOR = 8
_WIDEN_CAP = 2000

_ALLOWED_CONN_SQL = "SELECT session_id FROM messages WHERE id = ?"


class _EmptyCursor:
    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _SingleRowCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row] if self._row is not None else []


class _RestrictedConn:
    """Only the one lookup session_search performs directly on the connection.

    Handing over the live sqlite3.Connection would let any current or future
    code path in that module read the whole database.
    """

    def __init__(self, inner_conn, proxy: "ScopedSessionDB"):
        self._inner_conn = inner_conn
        self._proxy = proxy

    def execute(self, sql: str, params=()):
        normalised = " ".join((sql or "").split())
        if normalised != _ALLOWED_CONN_SQL:
            logger.error("scoped db: refused raw query: %s", normalised[:120])
            raise PermissionError("scoped session DB: query not allowed")

        row = self._inner_conn.execute(sql, params).fetchone()
        if row is None:
            return _EmptyCursor()

        try:
            session_id = row["session_id"] if hasattr(row, "keys") else row[0]
        except Exception:
            return _EmptyCursor()

        if not self._proxy.lineage_allowed(session_id):
            return _EmptyCursor()
        return _SingleRowCursor(row)


class ScopedSessionDB:
    def __init__(self, inner, scope: Scope):
        self._inner = inner
        self._scope = scope
        self._verdicts: Dict[str, bool] = {}

    # ---- gate --------------------------------------------------------

    def _row_allowed(self, row: Optional[Dict[str, Any]]) -> bool:
        if not row:
            return False
        chat_type = (row.get("chat_type") or "").strip()
        if not chat_type:
            return False
        chat_id = (row.get("chat_id") or "").strip()
        if chat_id and chat_id in self._scope.chat_ids:
            return True
        if chat_type == "dm":
            user_id = (row.get("user_id") or "").strip()
            if user_id and user_id in self._scope.user_ids:
                return True
        return False

    def lineage_allowed(self, session_id: str) -> bool:
        """Every ancestor must pass too.

        Compacted and delegated children inherit origin fields inconsistently,
        so requiring the whole chain is the conservative reading.
        """
        if not session_id:
            return False
        cached = self._verdicts.get(session_id)
        if cached is not None:
            return cached

        verdict = True
        seen = set()
        current = session_id
        while current and current not in seen:
            seen.add(current)
            try:
                row = self._inner.get_session(current)
            except Exception:
                logger.warning("scoped db: get_session failed for %s", current, exc_info=True)
                verdict = False
                break
            if not self._row_allowed(row):
                verdict = False
                break
            current = (row.get("parent_session_id") or "").strip()

        self._verdicts[session_id] = verdict
        return verdict

    def _filter_rows(self, rows: List[Dict[str, Any]], key: str, limit: Optional[int]):
        kept = [r for r in (rows or []) if self.lineage_allowed(str(r.get(key) or ""))]
        return kept[:limit] if limit else kept

    # ---- overridden reads --------------------------------------------

    def get_session(self, session_id: str, *args, **kwargs):
        # Returning None is the natural "not found" the callers already handle,
        # and it also stops the parent-walk in _resolve_to_parent.
        if not self.lineage_allowed(session_id):
            return None
        return self._inner.get_session(session_id, *args, **kwargs)

    def get_messages(self, session_id: str, *args, **kwargs):
        if not self.lineage_allowed(session_id):
            return []
        return self._inner.get_messages(session_id, *args, **kwargs)

    def get_messages_around(self, session_id: str, *args, **kwargs):
        if not self.lineage_allowed(session_id):
            return {"window": [], "messages_before": 0, "messages_after": 0}
        return self._inner.get_messages_around(session_id, *args, **kwargs)

    def get_anchored_view(self, session_id: str, *args, **kwargs):
        if not self.lineage_allowed(session_id):
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }
        return self._inner.get_anchored_view(session_id, *args, **kwargs)

    def search_messages(self, *args, **kwargs):
        limit = kwargs.get("limit")
        if isinstance(limit, int) and limit > 0:
            kwargs["limit"] = min(limit * _WIDEN_FACTOR, _WIDEN_CAP)
        rows = self._inner.search_messages(*args, **kwargs)
        return self._filter_rows(rows, "session_id", limit if isinstance(limit, int) else None)

    def list_sessions_rich(self, *args, **kwargs):
        limit = kwargs.get("limit")
        if isinstance(limit, int) and limit > 0:
            kwargs["limit"] = min(limit * _WIDEN_FACTOR, _WIDEN_CAP)
        rows = self._inner.list_sessions_rich(*args, **kwargs)
        return self._filter_rows(rows, "id", limit if isinstance(limit, int) else None)

    def resolve_session_by_title(self, *args, **kwargs):
        session_id = self._inner.resolve_session_by_title(*args, **kwargs)
        if session_id and self.lineage_allowed(session_id):
            return session_id
        return None

    @property
    def _conn(self):
        return _RestrictedConn(self._inner._conn, self)

    def close(self):
        # The inner handle is gateway-owned and shared across the process.
        return None

    def __getattr__(self, name: str):
        logger.error("scoped db: blocked access to unlisted attribute %r", name)
        raise AttributeError(
            f"{name!r} is not exposed by the scoped session DB "
            "(add an explicit, filtered override if it is needed)"
        )
