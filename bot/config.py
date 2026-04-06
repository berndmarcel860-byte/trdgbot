"""Configuration loader for trdgbot.

Reads ``config.yaml`` from the project root and merges in any environment
variables (for secrets) loaded from a ``.env`` file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and return the merged configuration dictionary.

    Args:
        config_path: Optional explicit path to a YAML config file.
                     Defaults to ``<project_root>/config.yaml``.

    Returns:
        Nested dictionary with all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    load_dotenv(_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else _ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    # Inject exchange credentials from environment variables
    exchange_cfg: Dict[str, Any] = cfg.setdefault("exchange", {})
    if os.environ.get("EXCHANGE_API_KEY"):
        exchange_cfg["api_key"] = os.environ["EXCHANGE_API_KEY"]
    if os.environ.get("EXCHANGE_API_SECRET"):
        exchange_cfg["api_secret"] = os.environ["EXCHANGE_API_SECRET"]
    if os.environ.get("EXCHANGE_PASSPHRASE"):
        exchange_cfg["passphrase"] = os.environ["EXCHANGE_PASSPHRASE"]

    return cfg


def get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from the config dictionary.

    Args:
        cfg: The configuration dictionary.
        *keys: Sequence of keys forming the path to the value.
        default: Value returned when the key path is not found.

    Returns:
        The value at the specified path or *default*.

    Example::

        get(cfg, "risk", "max_risk_per_trade", default=0.01)
    """
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is default:
            return default
    return node
