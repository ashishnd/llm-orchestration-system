"""Prompt registry.

Each agent registers its prompts here at import time using a stable key
of the form `<agent_name>.<role>` (e.g. `decomposition.system`,
`retrieval.user_template`). The meta-agent reads this registry to identify
the worst-performing prompt and propose a rewrite, and the orchestrator
reads it to load prompts at runtime.

Design notes:
- Decentralized definition (each agent file owns its prompts) but centralized
  lookup (one dict). Keeps prompts close to the code that uses them while
  giving the meta-agent a single index to scan.
- Stable keys are the contract. The meta-agent persists rewrite proposals
  by key; if a key changes, prior proposals lose their target.
- Prompts are plain strings, not Jinja templates. Variable substitution is
  done by `.format(**kwargs)` in each agent. Templates would add a
  dependency for very little gain at our scale.
- Approved rewrites override the registry at runtime via
  `set_active_prompt(key, text)`. The original ("baseline") version is kept
  in `_BASELINE` so we can diff and roll back.
"""

from __future__ import annotations

from threading import Lock

_BASELINE: dict[str, str] = {}
_ACTIVE: dict[str, str] = {}
_LOCK = Lock()


def register_prompt(key: str, text: str) -> None:
    """Register the baseline version of a prompt. Called at import time by
    each agent module. Idempotent — re-registering the same key/text is a
    no-op; re-registering with different text raises (the registry should
    be stable across the process)."""
    with _LOCK:
        if key in _BASELINE:
            if _BASELINE[key] != text:
                raise ValueError(
                    f"Prompt key {key!r} re-registered with different text. "
                    "The baseline registry should be append-only."
                )
            return
        _BASELINE[key] = text
        _ACTIVE[key] = text


def get_prompt(key: str) -> str:
    """Return the active prompt for `key`. Raises KeyError if unknown."""
    try:
        return _ACTIVE[key]
    except KeyError as e:
        raise KeyError(f"Unknown prompt key {key!r}. Known keys: {sorted(_ACTIVE.keys())}") from e


def get_baseline(key: str) -> str:
    """Return the original baseline prompt for diffing against rewrites."""
    return _BASELINE[key]


def set_active_prompt(key: str, text: str) -> str:
    """Override the active prompt for `key`. Returns the previously-active
    text so the caller can store it (for rollback). Used by the meta-agent
    flow after a human approves a rewrite proposal."""
    with _LOCK:
        if key not in _BASELINE:
            raise KeyError(f"Cannot set unknown prompt key {key!r}")
        previous = _ACTIVE[key]
        _ACTIVE[key] = text
        return previous


def reset_to_baseline(key: str) -> None:
    """Roll back an approved rewrite. Useful for the eval harness's targeted
    re-eval flow if a rewrite degrades performance."""
    with _LOCK:
        if key not in _BASELINE:
            raise KeyError(f"Cannot reset unknown prompt key {key!r}")
        _ACTIVE[key] = _BASELINE[key]


def all_keys() -> list[str]:
    """All registered prompt keys, sorted. The meta-agent uses this to
    enumerate candidates."""
    return sorted(_BASELINE.keys())


def snapshot() -> dict[str, str]:
    """Return a copy of the current active prompts. Useful for traces."""
    return dict(_ACTIVE)
