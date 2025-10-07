"""Utilities to enrich local PHC datasets from open data services."""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FACILITY_DATA = Path("data/phc_facility_status.csv")
DISEASE_DATA = Path("data/disease_incidence.csv")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class WorldBankIndicator:
    country: str
    indicator: str
    start_year: int
    end_year: int


@dataclass
class DHIS2Config:
    base_url: str
    username: str
    password: str
    verify_ssl: bool = True


class OpenDataRefresher:
    """Pulls fresh metrics from World Bank, WHO, and DHIS2 endpoints."""

    def __init__(self, session: Optional[requests.Session] = None, cache_dir: Path = CACHE_DIR) -> None:
        self.session = session or requests.Session()
        self.cache_dir = cache_dir

    # ---------------------------------------------------------------------
    # World Bank
    # ------------------------------------------------------------------
    def fetch_world_bank_indicator(self, request: WorldBankIndicator) -> List[Dict[str, object]]:
        """Fetch time series for a given World Bank indicator."""

        url = f"https://api.worldbank.org/v2/country/{request.country}/indicator/{request.indicator}"
        params = {
            "format": "json",
            "per_page": 2000,
            "date": f"{request.start_year}:{request.end_year}",
        }
        cache_file = self.cache_dir / f"worldbank_{request.country}_{request.indicator}.json"
        try:
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("World Bank API failed (%s), falling back to cache", exc)
            if cache_file.exists():
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                raise

        if not isinstance(payload, list) or len(payload) < 2:
            return []

        series = payload[1]
        cleaned: List[Dict[str, object]] = []
        for row in series:
            value = row.get("value")
            if value is None:
                continue
            cleaned.append(
                {
                    "country": row.get("countryiso3code", request.country).upper(),
                    "indicator": request.indicator,
                    "date": row.get("date"),
                    "value": float(value),
                }
            )
        return cleaned

    # ------------------------------------------------------------------
    # WHO Global Health Observatory
    # ------------------------------------------------------------------
    def fetch_who_gho_series(
        self,
        dimension: str,
        code: str,
        target_countries: Iterable[str],
    ) -> List[Dict[str, object]]:
        """Fetch indicator values from WHO GHO API for the provided countries."""

        base_url = "https://ghoapi.azureedge.net/api"
        cache_file = self.cache_dir / f"who_{dimension}_{code}.json"
        try:
            response = self.session.get(f"{base_url}/{dimension}('{code}')/Data", timeout=20)
            response.raise_for_status()
            payload = response.json()
            cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("WHO API failed (%s), falling back to cache", exc)
            if cache_file.exists():
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                raise

        results: List[Dict[str, object]] = []
        for row in payload.get("value", []):
            country = row.get("SpatialDim")
            if country and country.upper() in {code.upper() for code in target_countries}:
                try:
                    value = float(row.get("Value"))
                except (TypeError, ValueError):
                    continue
                results.append(
                    {
                        "country": country.upper(),
                        "indicator": code,
                        "date": row.get("TimeDim"),
                        "value": value,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # DHIS2 Analytics
    # ------------------------------------------------------------------
    def fetch_dhis2_analytics(
        self,
        config: DHIS2Config,
        dx: str,
        ou: str,
        period: str,
    ) -> Dict[str, object]:
        """Query a DHIS2 analytics endpoint."""

        url = f"{config.base_url.rstrip('/')}/api/analytics"
        params = {"dimension": [f"dx:{dx}", f"ou:{ou}", f"pe:{period}"], "displayProperty": "NAME"}
        try:
            response = self.session.get(
                url,
                params=params,
                auth=(config.username, config.password),
                timeout=30,
                verify=config.verify_ssl,
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("DHIS2 analytics request failed: %s", exc)
            raise
        return response.json()

    # ------------------------------------------------------------------
    # OpenHIE FHIR (Location/Encounter) sampling
    # ------------------------------------------------------------------
    def fetch_fhir_bundle(
        self,
        base_url: str,
        resource_type: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """Fetch resources from an OpenHIE-aligned FHIR endpoint."""

        url = f"{base_url.rstrip('/')}/{resource_type}"
        try:
            response = self.session.get(url, params=params or {}, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("FHIR request failed: %s", exc)
            raise
        return response.json()

    # ------------------------------------------------------------------
    # Dataset hydration helpers
    # ------------------------------------------------------------------
    def update_facility_dataset(self, facilities: List[Dict[str, object]], target: Path = FACILITY_DATA) -> None:
        """Persist refreshed facility metrics to CSV."""

        fieldnames = list(facilities[0].keys()) if facilities else []
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(facilities)

    def update_disease_dataset(self, disease_rows: List[Dict[str, object]], target: Path = DISEASE_DATA) -> None:
        """Persist refreshed disease incidence metrics to CSV."""

        fieldnames = list(disease_rows[0].keys()) if disease_rows else []
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disease_rows)

    # ------------------------------------------------------------------
    # Composite workflows
    # ------------------------------------------------------------------
    def refresh_from_world_bank(self, indicators: Iterable[WorldBankIndicator]) -> Dict[str, List[Dict[str, object]]]:
        """Fetch multiple World Bank indicators for downstream feature engineering."""

        refreshed: Dict[str, List[Dict[str, object]]] = {}
        for request in indicators:
            refreshed[request.indicator] = self.fetch_world_bank_indicator(request)
            time.sleep(0.2)  # be gentle with the public API
        return refreshed

    def refresh_all(
        self,
        indicators: Iterable[WorldBankIndicator],
        who_dimensions: Iterable[tuple[str, str]],
        dhis2_jobs: Iterable[tuple[DHIS2Config, str, str, str]],
    ) -> Dict[str, object]:
        """Run a full refresh cycle across supported providers."""

        payload: Dict[str, object] = {
            "world_bank": self.refresh_from_world_bank(indicators),
            "who": {},
            "dhis2": [],
        }

        for dimension, code in who_dimensions:
            payload["who"][code] = self.fetch_who_gho_series(dimension, code, [i.country for i in indicators])
            time.sleep(0.1)

        for config, dx, ou, period in dhis2_jobs:
            payload["dhis2"].append(
                {
                    "dx": dx,
                    "ou": ou,
                    "period": period,
                    "payload": self.fetch_dhis2_analytics(config, dx, ou, period),
                }
            )
            time.sleep(0.1)

        return payload


def _build_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Refresh PHC datasets from open data sources.")
    parser.add_argument("--country", default="NGA", help="Country ISO3 code for World Bank sampling")
    parser.add_argument(
        "--indicator",
        default="SH.MED.NUMW.P3",
        help="World Bank indicator ID (default: SH.MED.NUMW.P3 - nurses & midwives)",
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2022)
    args = parser.parse_args()

    refresher = OpenDataRefresher()
    indicators = [
        WorldBankIndicator(
            country=args.country,
            indicator=args.indicator,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    ]

    payload = refresher.refresh_from_world_bank(indicators)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI utility
    _build_cli()
