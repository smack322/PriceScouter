# backend/telemetry.py
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("pricescouter.telemetry")


def log_chart_event(event_name: str, **properties: Any) -> None:
    """
    Lightweight analytics hook.

    For now, logs to the server log. Later, can be wired to a DB table, GA, etc.
    """
    payload: Dict[str, Any] = {"event": event_name, **properties}
    logger.info("chart_event: %s", payload)
    # Placeholder for future:
    # save_analytics_event("chart", payload)
