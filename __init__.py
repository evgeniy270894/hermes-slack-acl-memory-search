"""ACL-scoped session_search for a shared Slack deployment.

Two mechanisms, both required:

1. Registry registration — owns the schema the model sees and produces the
   audit log line. On its own it is cosmetic: session_search belongs to
   _AGENT_LOOP_TOOLS and is intercepted by name in the agent loop, so registry
   dispatch never runs for it.

2. Module rebinding — agent/tool_executor.py and agent/agent_runtime_helpers.py
   both do `from tools.session_search_tool import session_search` *inside* the
   call, re-resolving the attribute every time. Rebinding it is what actually
   gates execution.
"""

from __future__ import annotations

import logging

from . import acl, action_token, selfcheck, slack_tool, tool

logger = logging.getLogger(__name__)

_ORIGINAL = None


def tool_handler(args: dict, **kwargs) -> str:
    """Registry entry point.

    Must be DEFINED here, in the package root. Hermes authorises an override by
    `handler.__globals__["__name__"]` (tools/registry.py:_plugin_owner_of) but
    registers the permission against the package root only. A handler declared
    in a submodule resolves to `<pkg>.tool`, finds no policy, and the override
    is rejected — with a message that points at config keys which are in fact
    already correct.
    """
    return tool.handler(args, **kwargs)


def slack_tool_handler(args: dict, **kwargs) -> str:
    """Registry entry point for the Slack read/search tool.

    A brand-new tool name needs no override permission (registry.register only
    consults the plugin-override policy when the name already exists under a
    different toolset), so unlike session_search this one is not forced to live
    in the package root. It is declared here anyway, next to its sibling, so
    both entry points are found in one place.
    """
    return slack_tool.handler(args, **kwargs)


def _install_module_patch() -> None:
    """Idempotent: plugin modules are re-executed on discover(force=True)."""
    global _ORIGINAL
    import tools.session_search_tool as sst

    current = sst.session_search
    if getattr(current, "__hermes_acl_wrapped__", False):
        return

    _ORIGINAL = current
    tool.set_original(current)
    sst.session_search = tool.scoped_session_search
    logger.info("hermes-slack-acl-memory-search: session_search rebound to scoped wrapper")


def register(ctx) -> None:
    from tools.registry import registry

    # Remove first, install second. If anything below raises, Hermes disables
    # the plugin and the tool is simply absent — which is the correct failure
    # mode. Registering over the built-in without removing it would leave the
    # unfiltered version live on any partial failure.
    registry.deregister("session_search")

    ctx.register_tool(
        name="session_search",
        toolset=selfcheck.EXPECTED_TOOLSET,  # distinct name so the override is audit-logged
        schema=tool.build_schema(),
        handler=tool_handler,
        check_fn=tool.check,
        emoji="🔒",
        description="Search past conversations you are entitled to see.",
        override=True,
    )

    ctx.register_tool(
        name=slack_tool.TOOL_NAME,
        toolset=slack_tool.TOOLSET,
        schema=slack_tool.build_schema(),
        handler=slack_tool_handler,
        check_fn=slack_tool.check,
        emoji="💬",
        description="Read and search Slack, scoped to what the asker may already see.",
    )
    slack_tool.mark_untrusted()

    _install_module_patch()
    ctx.register_hook("pre_gateway_dispatch", acl.capture_gateway)
    ctx.register_hook("pre_gateway_dispatch", action_token.capture)
    selfcheck.assert_installed()
