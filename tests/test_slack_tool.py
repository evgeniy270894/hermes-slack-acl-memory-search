"""Scope enforcement for the Slack read/search tool.

Callers are impersonated by patching the session-env lookup, which is the only
thing the resolver trusts. No Slack calls are made — the client is a fake.
"""

import importlib
import json
import os
import sys
import types
import unittest
from unittest import mock

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = "hermes_slack_acl_memory_search"

if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [_PLUGIN_DIR]
    sys.modules[_PKG] = _pkg

slack_tool = importlib.import_module(f"{_PKG}.slack_tool")
action_token = importlib.import_module(f"{_PKG}.action_token")

CHANNEL_MINE = "C_MINE"
CHANNEL_OTHER = "C_OTHER"
DM = "D_U1"
USER = "U1"


class FakeResp(dict):
    pass


class FakeClient:
    """Records calls; returns one message so a success is distinguishable."""

    def __init__(self):
        self.calls = []

    def conversations_history(self, **kw):
        self.calls.append(("history", kw))
        return FakeResp(ok=True, messages=[{"ts": "1.0", "user": USER, "text": "hello"}])

    def conversations_replies(self, **kw):
        self.calls.append(("replies", kw))
        return FakeResp(ok=True, messages=[{"ts": "1.0", "user": USER, "text": "reply"}])

    def api_call(self, api_path, json=None, **kw):
        # slack_sdk 3.43.0 has no binding for assistant.search.context, so the
        # tool goes through the generic escape hatch — mirror that here.
        self.calls.append(("search", {"api_path": api_path, **(json or {})}))
        resp = FakeResp(ok=True, results={"messages": []})
        resp.data = dict(resp)
        return resp

    def users_info(self, **kw):
        return FakeResp(ok=True, user={"profile": {"display_name": "Ann"}})


def run(args, *, chat_id, user_id=USER, member_channels=frozenset({CHANNEL_MINE})):
    env = {
        "HERMES_SESSION_PLATFORM": "slack",
        "HERMES_SESSION_CHAT_ID": chat_id,
        "HERMES_SESSION_USER_ID": user_id,
    }
    client = FakeClient()
    with mock.patch(f"{_PKG}.scope._session_env", side_effect=lambda n: env.get(n, "")), \
         mock.patch(f"{_PKG}.acl.channels_for_user", return_value=member_channels), \
         mock.patch.object(slack_tool, "_client", return_value=client):
        payload = json.loads(slack_tool.handler(args))
    return payload, client


class TestChannelScope(unittest.TestCase):
    def test_reads_its_own_channel(self):
        payload, client = run({"action": "read_channel"}, chat_id=CHANNEL_MINE)
        self.assertEqual(payload.get("channel"), CHANNEL_MINE)
        self.assertEqual(client.calls[0][0], "history")

    def test_defaults_to_the_current_channel(self):
        payload, _ = run({"action": "read_channel"}, chat_id=CHANNEL_MINE)
        self.assertEqual(payload["channel"], CHANNEL_MINE)

    def test_another_channel_is_refused_from_a_channel(self):
        payload, client = run(
            {"action": "read_channel", "channel": CHANNEL_OTHER}, chat_id=CHANNEL_MINE
        )
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])

    def test_a_dm_cannot_be_read_from_a_channel(self):
        payload, client = run({"action": "read_channel", "channel": DM}, chat_id=CHANNEL_MINE)
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])


class TestDmScope(unittest.TestCase):
    def test_member_channel_allowed_from_a_dm(self):
        payload, client = run({"action": "read_channel", "channel": CHANNEL_MINE}, chat_id=DM)
        self.assertEqual(payload.get("channel"), CHANNEL_MINE)
        self.assertEqual(client.calls[0][0], "history")

    def test_non_member_channel_refused_from_a_dm(self):
        payload, client = run({"action": "read_channel", "channel": CHANNEL_OTHER}, chat_id=DM)
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])

    def test_acl_failure_narrows_instead_of_widening(self):
        payload, client = run(
            {"action": "read_channel", "channel": CHANNEL_MINE},
            chat_id=DM,
            member_channels=None,  # resolve_scope treats None as "narrow to this chat"
        )
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])


class TestFailClosed(unittest.TestCase):
    def _blank_env(self, **over):
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": CHANNEL_MINE,
            "HERMES_SESSION_USER_ID": USER,
        }
        env.update(over)
        return env

    def _call(self, env):
        client = FakeClient()
        with mock.patch(f"{_PKG}.scope._session_env", side_effect=lambda n: env.get(n, "")), \
             mock.patch(f"{_PKG}.acl.channels_for_user", return_value=frozenset()), \
             mock.patch.object(slack_tool, "_client", return_value=client):
            return json.loads(slack_tool.handler({"action": "read_channel"})), client

    def test_missing_user_refuses(self):
        payload, client = self._call(self._blank_env(HERMES_SESSION_USER_ID=""))
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])

    def test_non_slack_platform_refuses(self):
        payload, client = self._call(self._blank_env(HERMES_SESSION_PLATFORM="cli"))
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])

    def test_unknown_action_refuses(self):
        payload, _ = run({"action": "delete_everything"}, chat_id=CHANNEL_MINE)
        self.assertIn("error", payload)


class TestSearch(unittest.TestCase):
    def setUp(self):
        action_token.clear_cache()

    def test_search_without_action_token_explains_itself(self):
        payload, client = run({"action": "search", "query": "x"}, chat_id=CHANNEL_MINE)
        self.assertIn("action_token", payload["error"])
        self.assertEqual(client.calls, [])

    def test_search_uses_the_captured_token(self):
        event = types.SimpleNamespace(
            raw_message={"action_token": "xoxa-test", "channel": CHANNEL_MINE, "user": USER}
        )
        action_token.capture(event=event)
        payload, client = run({"action": "search", "query": "x"}, chat_id=CHANNEL_MINE)
        self.assertEqual(payload.get("count"), 0)
        self.assertEqual(client.calls[0][0], "search")
        self.assertEqual(client.calls[0][1]["action_token"], "xoxa-test")
        self.assertEqual(client.calls[0][1]["api_path"], "assistant.search.context")

    def test_token_is_not_shared_across_users(self):
        event = types.SimpleNamespace(
            raw_message={"action_token": "xoxa-test", "channel": CHANNEL_MINE, "user": USER}
        )
        action_token.capture(event=event)
        self.assertIsNone(action_token.get(CHANNEL_MINE, "U_SOMEONE_ELSE"))


class TestNameResolution(unittest.TestCase):
    def test_a_name_resolves_and_is_then_authorised(self):
        with mock.patch.object(slack_tool, "_resolve_name", return_value=CHANNEL_MINE):
            payload, client = run({"action": "read_channel", "channel": "#mine"}, chat_id=DM)
        self.assertEqual(payload.get("channel"), CHANNEL_MINE)
        self.assertEqual(client.calls[0][0], "history")

    def test_resolution_does_not_bypass_scope(self):
        """A resolvable name still has to survive the membership check."""
        with mock.patch.object(slack_tool, "_resolve_name", return_value=CHANNEL_OTHER):
            payload, client = run({"action": "read_channel", "channel": "#secret"}, chat_id=DM)
        self.assertIn("error", payload)
        self.assertEqual(client.calls, [])

    def test_unknown_name_is_refused_the_same_way_as_a_forbidden_one(self):
        """Distinguishing them would leak which channels the bot sits in."""
        with mock.patch.object(slack_tool, "_resolve_name", return_value=None):
            unknown, _ = run({"action": "read_channel", "channel": "#nope"}, chat_id=DM)
        with mock.patch.object(slack_tool, "_resolve_name", return_value=CHANNEL_OTHER):
            forbidden, _ = run({"action": "read_channel", "channel": "#secret"}, chat_id=DM)
        self.assertEqual(unknown["error"], forbidden["error"])


class TestListChannels(unittest.TestCase):
    def test_lists_only_in_scope_channels_and_folds_thread_rows(self):
        directory = [
            {"id": CHANNEL_MINE, "name": "mine"},
            {"id": f"{CHANNEL_MINE}:1785.1", "name": "mine / topic"},
            {"id": CHANNEL_OTHER, "name": "secret"},
        ]
        with mock.patch.object(slack_tool, "_directory", return_value=directory):
            payload, client = run({"action": "list_channels"}, chat_id=DM)
        self.assertEqual([c["id"] for c in payload["channels"]], [CHANNEL_MINE])
        self.assertEqual(client.calls, [])


class TestUntrustedMarking(unittest.TestCase):
    def test_tool_name_is_added_to_the_untrusted_set(self):
        helpers = types.ModuleType("agent.tool_dispatch_helpers")
        helpers._UNTRUSTED_TOOL_NAMES = frozenset({"web_search"})
        parent = types.ModuleType("agent")
        parent.tool_dispatch_helpers = helpers
        # `import agent.tool_dispatch_helpers as x` binds through the parent
        # package, so both entries are required.
        with mock.patch.dict(
            sys.modules, {"agent": parent, "agent.tool_dispatch_helpers": helpers}
        ):
            slack_tool.mark_untrusted()
            self.assertIn(slack_tool.TOOL_NAME, helpers._UNTRUSTED_TOOL_NAMES)


if __name__ == "__main__":
    unittest.main()
