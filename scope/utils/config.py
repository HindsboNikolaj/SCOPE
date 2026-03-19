"""
config.py — YAML configuration loader for SCOPE.

Resolves ${ENV_VAR} references in YAML values to environment variables.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} patterns in strings."""
    if isinstance(value, str):
        def _replace(m):
            return os.environ.get(m.group(1), "")
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and resolve environment variable references."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _resolve_env(raw or {})


def load_agent_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load the SCOPE agent configuration.

    Searches in order:
      1. Explicit config_path argument
      2. SCOPE_CONFIG environment variable
      3. configs/agent_config.yaml relative to project root
    """
    if config_path:
        return load_config(config_path)

    env_path = os.getenv("SCOPE_CONFIG")
    if env_path:
        return load_config(env_path)

    project_root = Path(__file__).resolve().parents[2]
    default = project_root / "configs" / "agent_config.yaml"
    if default.exists():
        return load_config(default)

    return {}
