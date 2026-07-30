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

from . import acl, selfcheck, tool

logger = logging.getLogger(__name__)

_ORIGINAL = None


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
        handler=tool.handler,
        check_fn=tool.check,
        emoji="🔒",
        description="Search past conversations you are entitled to see.",
        override=True,
    )

    _install_module_patch()
    ctx.register_hook("pre_gateway_dispatch", acl.capture_gateway)
    selfcheck.assert_installed()
