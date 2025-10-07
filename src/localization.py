"""Localization utilities for PHC insights and recommendations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

DEFAULT_LANGUAGE = "en"

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "headline": "=== PHC Snapshot ===",
        "no_priority_facilities": "All facilities meet minimum standards.",
        "priority_title": "Priority facilities",
        "recommendation_title": "Recommended actions",
        "sms_header": "PHC Alert",
        "staff_deployment": "Deploy additional staff",
        "medicine_restock": "Restock essential medicines",
        "referral_support": "Activate referral support",
        "surge_plan": "Open surge clinic",
    },
    "fr": {
        "headline": "=== Aperçu des SSP ===",
        "no_priority_facilities": "Toutes les formations respectent les normes minimales.",
        "priority_title": "Structures prioritaires",
        "recommendation_title": "Actions recommandées",
        "sms_header": "Alerte SSP",
        "staff_deployment": "Déployer du personnel",
        "medicine_restock": "Réapprovisionner les médicaments essentiels",
        "referral_support": "Activer le soutien aux références",
        "surge_plan": "Ouvrir une clinique de débordement",
    },
    "sw": {
        "headline": "=== Muhtasari wa Huduma ya Msingi ===",
        "no_priority_facilities": "Vituo vyote vinatimiza viwango vya chini.",
        "priority_title": "Vituo vya kipaumbele",
        "recommendation_title": "Hatua zilizopendekezwa",
        "sms_header": "Tahadhari ya PHC",
        "staff_deployment": "Tuma watoa huduma zaidi",
        "medicine_restock": "Jaza tena dawa muhimu",
        "referral_support": "Washa msaada wa rufaa",
        "surge_plan": "Fungua kituo cha msongamano",
    },
    "pt": {
        "headline": "=== Panorama da APS ===",
        "no_priority_facilities": "Todas as unidades cumprem os padrões mínimos.",
        "priority_title": "Unidades prioritárias",
        "recommendation_title": "Ações recomendadas",
        "sms_header": "Alerta APS",
        "staff_deployment": "Mobilizar mais profissionais",
        "medicine_restock": "Reforçar estoque de medicamentos",
        "referral_support": "Ativar suporte de referência",
        "surge_plan": "Abrir clínica de contingência",
    },
}


@dataclass
class LocalisedPackage:
    """Container for localised narrative assets."""

    headline: str
    summary_lines: List[str]
    priority_lines: List[str]
    recommendation_lines: List[str]
    sms_digest: str
    no_priority_facilities: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "headline": self.headline,
            "summary_lines": self.summary_lines,
            "priority_lines": self.priority_lines,
            "recommendation_lines": self.recommendation_lines,
            "sms_digest": self.sms_digest,
            "no_priority_facilities": self.no_priority_facilities,
        }


def _translate(key: str, language: str) -> str:
    language_map = TRANSLATIONS.get(language, TRANSLATIONS[DEFAULT_LANGUAGE])
    return language_map.get(key, TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key))


def available_languages() -> Sequence[str]:
    """Return supported language codes."""

    return tuple(sorted(TRANSLATIONS.keys()))


def _format_country_summary(country: Dict[str, object], language: str) -> str:
    return (
        f"{country['country']}: {country['facilities']} facilities, "
        f"staff/10k {country['avg_staff_per_10k']:.1f}, "
        f"functionality {country['avg_functionality']:.2f}, "
        f"stock-outs {country['stockout_rate']:.2f}"
    )


def _format_recommendation(rec: Dict[str, object], language: str) -> str:
    actions: List[str] = []
    if rec["required_staff"] > 0:
        actions.append(_translate("staff_deployment", language))
    if rec["urgent_stock_replenishment"]:
        actions.append(_translate("medicine_restock", language))
    if rec["refer_patients"]:
        actions.append(_translate("referral_support", language))
    if rec.get("surge_alert"):
        actions.append(_translate("surge_plan", language))
    action_text = ", ".join(actions) if actions else "-"
    return (
        f"{rec['country']}·{rec['facility_id']}: {action_text}. "
        f"Triage: {rec['triage_guideline']}"
    )


def localise_insights(insights: Dict[str, object], language: str) -> Dict[str, object]:
    """Localise narrative elements for dashboards, SMS, or chatbots."""

    translations = TRANSLATIONS.get(language, TRANSLATIONS[DEFAULT_LANGUAGE])

    summary_lines = [
        _format_country_summary(country, language)
        for country in insights["country_summary"]
    ]

    priority_lines = [
        f"{facility['country']}·{facility['facility_id']}: staff/10k {facility['staff_per_10k']:.1f}, "
        f"functionality {facility['functionality_score']:.2f}"
        for facility in insights["underserved_facilities"]
    ]

    recommendation_lines = [
        _format_recommendation(rec, language) for rec in insights["recommendations"]
    ]

    sms_body = " | ".join(summary_lines[:2])
    sms_digest = f"{translations['sms_header']}: {sms_body}" if sms_body else translations["sms_header"]

    package = LocalisedPackage(
        headline=translations["headline"],
        summary_lines=summary_lines,
        priority_lines=priority_lines,
        recommendation_lines=recommendation_lines,
        sms_digest=sms_digest,
        no_priority_facilities=translations["no_priority_facilities"],
    )
    return package.as_dict()


__all__ = ["available_languages", "localise_insights", "DEFAULT_LANGUAGE"]
