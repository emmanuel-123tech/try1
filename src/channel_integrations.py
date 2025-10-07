"""Messaging and low-bandwidth channel integrations for PHC insights."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - allow execution as script
    from . import localization
except ImportError:  # pragma: no cover
    import localization  # type: ignore

DEFAULT_LANGUAGE = localization.DEFAULT_LANGUAGE


def build_rapidpro_flow(insights: Dict[str, object], language: str = DEFAULT_LANGUAGE) -> Dict[str, object]:
    """Create a lightweight RapidPro flow presenting headline PHC insights."""

    localised = localization.localise_insights(insights, language)
    messages: List[Dict[str, object]] = []

    for line in localised.get("summary_lines", [])[:3]:
        messages.append({"type": "send_msg", "text": line})

    if not localised.get("priority_lines"):
        messages.append({"type": "send_msg", "text": localised.get("no_priority_facilities", "")})
    else:
        for priority in localised.get("priority_lines", [])[:3]:
            messages.append({"type": "send_msg", "text": priority})

    for rec in localised.get("recommendation_lines", [])[:3]:
        messages.append({"type": "send_msg", "text": rec})

    return {
        "flow": {
            "name": "PHC Insight Broadcast",
            "language": language,
            "nodes": [{"actions": messages}],
        }
    }


def format_ussd_menu(insights: Dict[str, object], language: str = DEFAULT_LANGUAGE) -> str:
    """Format a compact USSD-friendly menu of insights."""

    localised = localization.localise_insights(insights, language)
    lines: List[str] = [localised.get("headline", "PHC")]

    for country_line in localised.get("summary_lines", [])[:2]:
        lines.append(country_line[:120])

    if localised.get("priority_lines"):
        lines.append("1. " + localised["priority_lines"][0][:120])
    else:
        lines.append("1. " + localised.get("no_priority_facilities", ""))

    lines.append("2. " + (localised.get("recommendation_lines", [""])[0][:120] if localised.get("recommendation_lines") else "-"))
    lines.append("#. Exit")
    return "\n".join(lines)


def export_channel_assets(
    insights: Dict[str, object],
    target: Path,
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Persist RapidPro and USSD payloads for downstream integration."""

    target.mkdir(parents=True, exist_ok=True)
    rapidpro_payload = build_rapidpro_flow(insights, language)
    (target / "rapidpro_flow.json").write_text(json.dumps(rapidpro_payload, indent=2), encoding="utf-8")
    (target / "ussd_menu.txt").write_text(format_ussd_menu(insights, language), encoding="utf-8")
    (target / "sms_digest.txt").write_text(
        localization.localise_insights(insights, language).get("sms_digest", ""),
        encoding="utf-8",
    )


__all__ = ["build_rapidpro_flow", "format_ussd_menu", "export_channel_assets", "DEFAULT_LANGUAGE"]
