"""Interpret simulation-template support settings.

The detector interval and the wavelength range supplied by a source template
are different quantities. This module keeps that distinction in one place.
"""

from __future__ import annotations

from typing import Any


def template_support_policy(observation: dict[str, Any]) -> str:
    """Return ``complete`` or ``retain``, including the former setting name."""
    configured = observation.get("template_support_policy")
    legacy = observation.get("require_complete_wavelength_coverage")
    if configured is None:
        return "complete" if bool(legacy) else "retain"
    policy = str(configured)
    if policy not in {"complete", "retain"}:
        raise ValueError("template_support_policy must be 'complete' or 'retain'")
    if legacy is not None and bool(legacy) != (policy == "complete"):
        raise ValueError(
            "template_support_policy conflicts with require_complete_wavelength_coverage"
        )
    return policy
