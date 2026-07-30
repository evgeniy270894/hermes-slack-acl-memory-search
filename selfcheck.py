"""Refuse to run at all if the scoped tool is not actually installed.

Hermes catches plugin exceptions and carries on with the built-in tool, so a
broken install would silently serve unfiltered search. This turns that into a
hard stop.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import List

logger = logging.getLogger(__name__)

EXPECTED_TOOLSET = "slack_acl_session_search"

# The whole interception rests on these two call sites late-importing
# session_search on every call. An upgrade that hoists the import to module
# level would bypass our patch without any other visible symptom.
_LATE_IMPORT_MARKER = "from tools.session_search_tool import session_search"
_INTERCEPTED = (
    ("agent.tool_executor", "execute_tool_calls_sequential"),
    ("agent.agent_runtime_helpers", "invoke_tool"),
)


def _check_registry(problems: List[str]) -> None:
    from tools.registry import registry

    import hermes_plugins.hermes_slack_acl_memory_search as pkg

    entry = registry.get_entry("session_search")
    if entry is None:
        problems.append("session_search is not registered")
        return
    if getattr(entry, "handler", None) is not pkg.tool_handler:
        problems.append("registry handler is not ours")
    if getattr(entry, "toolset", None) != EXPECTED_TOOLSET:
        problems.append(f"registry toolset is {getattr(entry, 'toolset', None)!r}")


def _check_module_patch(problems: List[str]) -> None:
    import tools.session_search_tool as sst

    if getattr(sst.session_search, "__hermes_acl_wrapped__", False) is not True:
        problems.append("tools.session_search_tool.session_search is not the ACL wrapper")


def _check_interception_points(problems: List[str]) -> None:
    for module_name, func_name in _INTERCEPTED:
        try:
            module = __import__(module_name, fromlist=[func_name])
            source = inspect.getsource(getattr(module, func_name))
        except Exception as exc:
            problems.append(f"cannot inspect {module_name}.{func_name}: {exc}")
            continue
        if _LATE_IMPORT_MARKER not in source:
            problems.append(
                f"{module_name}.{func_name} no longer late-imports session_search — "
                "the module patch would be bypassed"
            )


def assert_installed() -> None:
    problems: List[str] = []
    _check_registry(problems)
    _check_module_patch(problems)
    _check_interception_points(problems)

    if not problems:
        logger.info("hermes-slack-acl-memory-search: scoped session_search verified")
        return

    message = "hermes-slack-acl-memory-search: FATAL — " + "; ".join(problems)
    logger.critical(message)
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except Exception:
        pass

    # os._exit rather than SystemExit: Hermes catches Exception around plugin
    # loading, and SystemExit could still be swallowed unwinding through
    # asyncio. Serving unfiltered search is not an acceptable fallback.
    os._exit(70)  # EX_SOFTWARE
