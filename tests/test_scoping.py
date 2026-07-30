"""Scope and proxy tests. Run without Hermes installed:

    python3 -m unittest discover tests -v
"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

# Load the plugin as a package, mirroring how Hermes imports it
# (hermes_plugins.<slug>). The directory name contains hyphens, so it cannot
# be imported directly.
_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = "hermes_slack_acl_memory_search"

if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [_PLUGIN_DIR]
    sys.modules[_PKG] = _pkg

ScopedSessionDB = importlib.import_module(f"{_PKG}.db_proxy").ScopedSessionDB
_scope_mod = importlib.import_module(f"{_PKG}.scope")
AskingContext = _scope_mod.AskingContext
Scope = _scope_mod.Scope
resolve_scope = _scope_mod.resolve_scope

CHANNEL_A = "C_AAA"
CHANNEL_B = "C_BBB"
DM_U1 = "D_U1"
DM_U2 = "D_U2"
USER_1 = "U1"
USER_2 = "U2"


class FakeDB:
    """Minimal stand-in for SessionDB with the columns the proxy inspects."""

    def __init__(self, sessions, messages=None):
        self._sessions = sessions
        self._messages = messages or []
        self.calls = []

    def get_session(self, session_id, *a, **kw):
        self.calls.append(("get_session", session_id))
        return self._sessions.get(session_id)

    def get_messages(self, session_id, *a, **kw):
        return [m for m in self._messages if m["session_id"] == session_id]

    def get_messages_around(self, session_id, *a, **kw):
        return {"window": [{"session_id": session_id}], "messages_before": 1, "messages_after": 1}

    def get_anchored_view(self, session_id, *a, **kw):
        return {
            "window": [{"session_id": session_id}],
            "messages_before": 1,
            "messages_after": 1,
            "bookend_start": [],
            "bookend_end": [],
        }

    def search_messages(self, *a, **kw):
        self.calls.append(("search_messages", kw.get("limit")))
        return list(self._messages)

    def list_sessions_rich(self, *a, **kw):
        return [{"id": sid} for sid in self._sessions]

    def resolve_session_by_title(self, title, *a, **kw):
        for sid, row in self._sessions.items():
            if row.get("title") == title:
                return sid
        return None


def session_row(chat_id, chat_type, user_id, parent=None, title=None):
    return {
        "chat_id": chat_id,
        "chat_type": chat_type,
        "user_id": user_id,
        "parent_session_id": parent,
        "title": title,
        "source": "slack",
    }


def build_db():
    return FakeDB(
        sessions={
            "s_chan_a": session_row(CHANNEL_A, "group", USER_1, title="alpha"),
            "s_chan_b": session_row(CHANNEL_B, "group", USER_2, title="beta"),
            "s_dm_u1": session_row(DM_U1, "dm", USER_1, title="gamma"),
            "s_dm_u2": session_row(DM_U2, "dm", USER_2, title="delta"),
            "s_cli": session_row(None, None, None, title="cli"),
            "s_child_ok": session_row(CHANNEL_A, "group", USER_1, parent="s_chan_a"),
            "s_child_bad": session_row(CHANNEL_A, "group", USER_1, parent="s_chan_b"),
        },
        messages=[
            {"session_id": "s_chan_a", "content": "secret"},
            {"session_id": "s_chan_b", "content": "secret"},
            {"session_id": "s_dm_u1", "content": "secret"},
            {"session_id": "s_dm_u2", "content": "secret"},
            {"session_id": "s_cli", "content": "secret"},
            {"session_id": "s_child_ok", "content": "secret"},
            {"session_id": "s_child_bad", "content": "secret"},
        ],
    )


class TestChannelScope(unittest.TestCase):
    def setUp(self):
        self.db = build_db()
        self.scope = Scope(frozenset({CHANNEL_A}), frozenset(), "channel")
        self.proxy = ScopedSessionDB(self.db, self.scope)

    def test_only_own_channel_rows_survive_search(self):
        rows = self.proxy.search_messages(limit=10)
        self.assertEqual({r["session_id"] for r in rows}, {"s_chan_a", "s_child_ok"})

    def test_other_channel_denied(self):
        self.assertIsNone(self.proxy.get_session("s_chan_b"))
        self.assertEqual(self.proxy.get_messages("s_chan_b"), [])

    def test_dm_denied_from_channel(self):
        self.assertIsNone(self.proxy.get_session("s_dm_u1"))

    def test_null_chat_type_denied(self):
        self.assertIsNone(self.proxy.get_session("s_cli"))

    def test_browse_scoped(self):
        rows = self.proxy.list_sessions_rich(limit=50)
        self.assertEqual({r["id"] for r in rows}, {"s_chan_a", "s_child_ok"})

    def test_title_lookup_scoped(self):
        self.assertEqual(self.proxy.resolve_session_by_title("alpha"), "s_chan_a")
        self.assertIsNone(self.proxy.resolve_session_by_title("beta"))


class TestDmScope(unittest.TestCase):
    def setUp(self):
        self.db = build_db()
        # U1 in their DM, member of channel A.
        self.scope = Scope(frozenset({DM_U1, CHANNEL_A}), frozenset({USER_1}), "dm")
        self.proxy = ScopedSessionDB(self.db, self.scope)

    def test_own_dm_and_member_channel_visible(self):
        rows = self.proxy.search_messages(limit=10)
        self.assertEqual({r["session_id"] for r in rows}, {"s_dm_u1", "s_chan_a", "s_child_ok"})

    def test_other_users_dm_denied(self):
        """The exact leak that was reproduced on the live server."""
        self.assertIsNone(self.proxy.get_session("s_dm_u2"))
        rows = self.proxy.search_messages(limit=10)
        self.assertNotIn("s_dm_u2", {r["session_id"] for r in rows})

    def test_non_member_channel_denied(self):
        self.assertIsNone(self.proxy.get_session("s_chan_b"))


class TestLineage(unittest.TestCase):
    def setUp(self):
        self.db = build_db()
        self.scope = Scope(frozenset({CHANNEL_A}), frozenset(), "channel")
        self.proxy = ScopedSessionDB(self.db, self.scope)

    def test_child_of_in_scope_parent_allowed(self):
        self.assertIsNotNone(self.proxy.get_session("s_child_ok"))

    def test_child_of_out_of_scope_parent_denied(self):
        # Own chat_id is in scope, but the ancestor is not.
        self.assertIsNone(self.proxy.get_session("s_child_bad"))


class TestWidening(unittest.TestCase):
    def test_search_limit_is_widened_then_truncated(self):
        db = build_db()
        proxy = ScopedSessionDB(db, Scope(frozenset({CHANNEL_A}), frozenset(), "channel"))
        proxy.search_messages(limit=3)
        widened = [c for c in db.calls if c[0] == "search_messages"][0][1]
        self.assertGreater(widened, 3)


class TestRestrictedConn(unittest.TestCase):
    class _Conn:
        def __init__(self, row):
            self._row = row

        def execute(self, sql, params=()):
            class _C:
                def __init__(self, row):
                    self._row = row

                def fetchone(self):
                    return self._row

            return _C(self._row)

    def _proxy_with_conn(self, row):
        db = build_db()
        db._conn = self._Conn(row)
        return ScopedSessionDB(db, Scope(frozenset({CHANNEL_A}), frozenset(), "channel"))

    def test_allowed_query_filters_out_of_scope(self):
        proxy = self._proxy_with_conn(("s_chan_b",))
        cur = proxy._conn.execute("SELECT session_id FROM messages WHERE id = ?", (1,))
        self.assertIsNone(cur.fetchone())

    def test_allowed_query_passes_in_scope(self):
        proxy = self._proxy_with_conn(("s_chan_a",))
        cur = proxy._conn.execute("SELECT session_id FROM messages WHERE id = ?", (1,))
        self.assertIsNotNone(cur.fetchone())

    def test_arbitrary_query_refused(self):
        proxy = self._proxy_with_conn(("s_chan_a",))
        with self.assertRaises(PermissionError):
            proxy._conn.execute("SELECT * FROM messages")


class TestUnknownAttribute(unittest.TestCase):
    def test_unlisted_method_raises(self):
        proxy = ScopedSessionDB(build_db(), Scope(frozenset(), frozenset(), "x"))
        with self.assertRaises(AttributeError):
            proxy.some_future_method


class TestScopeResolution(unittest.TestCase):
    def test_channel_scope_ignores_acl(self):
        ctx = AskingContext("slack", CHANNEL_A, USER_1, "group", "s1")
        acl = mock.Mock()
        scope = resolve_scope(ctx, acl)
        self.assertEqual(scope.chat_ids, frozenset({CHANNEL_A}))
        self.assertEqual(scope.user_ids, frozenset())
        acl.channels_for_user.assert_not_called()

    def test_dm_scope_includes_member_channels(self):
        ctx = AskingContext("slack", DM_U1, USER_1, "dm", "s1")
        acl = mock.Mock()
        acl.channels_for_user.return_value = frozenset({CHANNEL_A})
        scope = resolve_scope(ctx, acl)
        self.assertEqual(scope.chat_ids, frozenset({DM_U1, CHANNEL_A}))
        self.assertEqual(scope.user_ids, frozenset({USER_1}))

    def test_dm_scope_narrows_when_acl_unavailable(self):
        ctx = AskingContext("slack", DM_U1, USER_1, "dm", "s1")
        acl = mock.Mock()
        acl.channels_for_user.return_value = None
        scope = resolve_scope(ctx, acl)
        self.assertEqual(scope.chat_ids, frozenset({DM_U1}))
        self.assertEqual(scope.user_ids, frozenset())
        self.assertEqual(scope.reason, "dm-acl-unavailable")


if __name__ == "__main__":
    unittest.main()
