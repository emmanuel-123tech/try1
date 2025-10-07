"""Analytics toolkit for strengthening Primary Health Care (PHC) delivery.

The module ingests facility readiness indicators and disease incidence trends and
returns a unified package of:

* Coverage and service quality gaps
* Gradient-boosted surge forecasts enriched with structural trend priors
* Localised recommendations ready for dashboard, chatbot, or USSD delivery

It can be executed directly to print insights, export JSON, or power the
companion Streamlit and messaging interfaces.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - support execution as module or script
    from . import localization  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import localization  # type: ignore[import-not-found]

try:  # pragma: no cover - optional channel tooling
    from . import channel_integrations  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import channel_integrations  # type: ignore[import-not-found]


FACILITY_FILE = Path("data/phc_facility_status.csv")
DISEASE_FILE = Path("data/disease_incidence.csv")
DEFAULT_LANGUAGE = "en"


@dataclass
class ForecastResult:
    """Represents near-term forecasts for a specific region."""

    country: str
    region: str
    indicator: str
    current_value: float
    forecast_value: float
    delta: float
    confidence_interval: Tuple[float, float]

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "country": self.country,
            "region": self.region,
            "indicator": self.indicator,
            "current_value": round(self.current_value, 2),
            "forecast_value": round(self.forecast_value, 2),
            "delta": round(self.delta, 2),
            "confidence_interval": [
                round(self.confidence_interval[0], 2),
                round(self.confidence_interval[1], 2),
            ],
        }


@dataclass
class RegionContext:
    """Aggregated facility context used as exogenous drivers in forecasting."""

    avg_staff_per_10k: float
    avg_functionality: float
    avg_stockout: float
    daily_visits: float
    facilities: int


def load_facility_data(path: Path = FACILITY_FILE) -> List[Dict[str, object]]:
    """Load facility readiness data into a list of dictionaries."""

    facilities: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            facilities.append(
                {
                    "country": row["country"],
                    "region": row["region"],
                    "facility_id": row["facility_id"],
                    "facility_type": row["facility_type"],
                    "catchment_population": float(row["catchment_population"]),
                    "staff_count": float(row["staff_count"]),
                    "daily_visits": float(row["daily_visits"]),
                    "stockout_rate": float(row["stockout_rate"]),
                    "functionality_score": float(row["functionality_score"]),
                    "electricity_reliability": float(row["electricity_reliability"]),
                    "water_access": bool(int(row["water_access"])),
                }
            )
    return facilities


def load_disease_data(path: Path = DISEASE_FILE) -> List[Dict[str, object]]:
    """Load disease incidence data and enrich with timestamp metadata."""

    disease_rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            month = row["month"]
            month_dt = datetime.strptime(f"{month}-01", "%Y-%m-%d")
            disease_rows.append(
                {
                    "country": row["country"],
                    "region": row["region"],
                    "month": month,
                    "month_dt": month_dt,
                    "malaria_cases": float(row["malaria_cases"]),
                    "respiratory_cases": float(row["respiratory_cases"]),
                    "maternal_complications": float(row["maternal_complications"]),
                }
            )
    return disease_rows


def compute_facility_gaps(facilities: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Derive staffing and service quality indicators per facility."""

    enriched: List[Dict[str, object]] = []
    for facility in facilities:
        record = dict(facility)
        pop = facility["catchment_population"] or 1
        staff = facility["staff_count"]
        record["staff_per_10k"] = (staff / pop) * 10000
        record["visits_per_staff"] = (facility["daily_visits"] / staff) if staff else 0.0
        record["undercapacity"] = (
            record["staff_per_10k"] < 12
            or record["functionality_score"] < 0.6
            or not record["water_access"]
            or record["electricity_reliability"] < 0.6
        )
        enriched.append(record)
    return enriched


def _aggregate_region_context(
    facilities: Sequence[Dict[str, object]]
) -> Dict[Tuple[str, str], RegionContext]:
    """Summarise facility indicators for exogenous modelling features."""

    context_totals: Dict[Tuple[str, str], Dict[str, float]] = {}
    counts: Dict[Tuple[str, str], int] = {}

    for facility in facilities:
        key = (facility["country"], facility["region"])
        counts[key] = counts.get(key, 0) + 1
        bucket = context_totals.setdefault(
            key,
            {
                "staff": 0.0,
                "functionality": 0.0,
                "stockout": 0.0,
                "visits": 0.0,
            },
        )
        bucket["staff"] += facility["staff_per_10k"]
        bucket["functionality"] += facility["functionality_score"]
        bucket["stockout"] += facility["stockout_rate"]
        bucket["visits"] += facility["daily_visits"]

    contexts: Dict[Tuple[str, str], RegionContext] = {}
    for key, bucket in context_totals.items():
        denominator = counts[key]
        contexts[key] = RegionContext(
            avg_staff_per_10k=bucket["staff"] / denominator if denominator else 0.0,
            avg_functionality=bucket["functionality"] / denominator if denominator else 0.0,
            avg_stockout=bucket["stockout"] / denominator if denominator else 0.0,
            daily_visits=bucket["visits"] / denominator if denominator else 0.0,
            facilities=denominator,
        )

    return contexts


def aggregate_country_landscape(facilities: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Summarise key coverage gaps by country."""

    summary: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for facility in facilities:
        country = facility["country"]
        counts[country] = counts.get(country, 0) + 1
        bucket = summary.setdefault(
            country,
            {
                "staff_per_10k": 0.0,
                "functionality": 0.0,
                "stockout": 0.0,
                "undercapacity": 0.0,
            },
        )
        bucket["staff_per_10k"] += facility["staff_per_10k"]
        bucket["functionality"] += facility["functionality_score"]
        bucket["stockout"] += facility["stockout_rate"]
        bucket["undercapacity"] += 1.0 if facility["undercapacity"] else 0.0

    results: List[Dict[str, object]] = []
    for country, metrics in summary.items():
        total = counts[country]
        results.append(
            {
                "country": country,
                "facilities": total,
                "avg_staff_per_10k": metrics["staff_per_10k"] / total if total else 0.0,
                "avg_functionality": metrics["functionality"] / total if total else 0.0,
                "stockout_rate": metrics["stockout"] / total if total else 0.0,
                "undercapacity_rate": round((metrics["undercapacity"] / total) if total else 0.0, 2),
            }
        )
    return results


def _month_offset(timestamp: datetime, offset: int) -> datetime:
    """Return timestamp advanced by ``offset`` months."""

    year = timestamp.year + (timestamp.month - 1 + offset) // 12
    month = ((timestamp.month - 1 + offset) % 12) + 1
    return datetime(year=year, month=month, day=1)


def _build_time_features(
    months: Sequence[datetime],
    context: Optional[RegionContext],
    horizon: int = 1,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Create feature matrices for gradient boosting forecasts."""

    base_rows: List[List[float]] = []
    for index, month in enumerate(months):
        seasonal_position = (month.month - 1) / 12
        row = [
            float(index),
            float(month.year),
            float(month.month),
            math.sin(2 * math.pi * seasonal_position),
            math.cos(2 * math.pi * seasonal_position),
        ]
        if context:
            row.extend(
                [
                    context.avg_staff_per_10k,
                    context.avg_functionality,
                    context.avg_stockout,
                    context.daily_visits,
                ]
            )
        base_rows.append(row)

    future_rows: List[List[float]] = []
    last_index = len(months)
    last_month = months[-1] if months else datetime.utcnow()
    for step in range(horizon):
        future_month = _month_offset(last_month, step + 1)
        seasonal_position = (future_month.month - 1) / 12
        row = [
            float(last_index + step),
            float(future_month.year),
            float(future_month.month),
            math.sin(2 * math.pi * seasonal_position),
            math.cos(2 * math.pi * seasonal_position),
        ]
        if context:
            row.extend(
                [
                    context.avg_staff_per_10k,
                    context.avg_functionality,
                    context.avg_stockout,
                    context.daily_visits,
                ]
            )
        future_rows.append(row)

    return base_rows, future_rows


class DecisionStump:
    """Lightweight regression stump used as the base learner."""

    def __init__(self) -> None:
        self.feature_index: int = 0
        self.threshold: float = 0.0
        self.left_value: float = 0.0
        self.right_value: float = 0.0

    def fit(self, features: Sequence[Sequence[float]], residuals: Sequence[float]) -> None:
        if not features:
            raise ValueError("Cannot fit stump without features")

        best_loss = float("inf")
        best: Optional[Tuple[int, float, float, float]] = None

        n_features = len(features[0])
        for feature_index in range(n_features):
            column = [row[feature_index] for row in features]
            thresholds = sorted(set(column))
            for threshold in thresholds:
                left_values = [res for res, value in zip(residuals, column) if value <= threshold]
                right_values = [res for res, value in zip(residuals, column) if value > threshold]
                left_value = mean(left_values) if left_values else 0.0
                right_value = mean(right_values) if right_values else 0.0
                predictions = [
                    left_value if value <= threshold else right_value
                    for value in column
                ]
                loss = mean((res - pred) ** 2 for res, pred in zip(residuals, predictions))
                if loss < best_loss:
                    best_loss = loss
                    best = (
                        feature_index,
                        float(threshold),
                        float(left_value),
                        float(right_value),
                    )

        if best is None:
            avg = mean(residuals) if residuals else 0.0
            best = (0, 0.0, float(avg), float(avg))

        (
            self.feature_index,
            self.threshold,
            self.left_value,
            self.right_value,
        ) = best

    def predict(self, features: Sequence[Sequence[float]]) -> List[float]:
        predictions: List[float] = []
        for row in features:
            predictions.append(
                self.left_value if row[self.feature_index] <= self.threshold else self.right_value
            )
        return predictions


class GradientBoostingTimeSeries:
    """Gradient boosting ensemble tailored for small health time series."""

    def __init__(self, n_estimators: int = 75, learning_rate: float = 0.1) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_prediction: float = 0.0
        self.estimators: List[DecisionStump] = []

    def fit(self, features: Sequence[Sequence[float]], values: Sequence[float]) -> None:
        values_list = [float(v) for v in values]
        if not values_list:
            raise ValueError("Cannot fit gradient boosting model without observations")

        self.initial_prediction = float(mean(values_list))
        predictions = [self.initial_prediction for _ in values_list]

        for _ in range(self.n_estimators):
            residuals = [value - pred for value, pred in zip(values_list, predictions)]
            if all(abs(res) < 1e-6 for res in residuals):
                break
            stump = DecisionStump()
            stump.fit(features, residuals)
            update = stump.predict(features)
            predictions = [
                pred + self.learning_rate * delta
                for pred, delta in zip(predictions, update)
            ]
            self.estimators.append(stump)

    def predict(self, features: Sequence[Sequence[float]]) -> List[float]:
        predictions = [self.initial_prediction for _ in features]
        for stump in self.estimators:
            updates = stump.predict(features)
            predictions = [
                pred + self.learning_rate * delta
                for pred, delta in zip(predictions, updates)
            ]
        return predictions


def _structural_forecast(values: Sequence[float], horizon: int = 1) -> List[float]:
    """Fallback Bayesian structural forecast based on a local trend prior."""

    if not values:
        return [0.0 for _ in range(horizon)]

    mean_value = float(mean(values))
    variance = float(pvariance(values)) if len(values) > 1 else 0.0
    results: List[float] = []
    last = values[-1]
    rng = random.Random(42)

    for _ in range(horizon):
        trend = last + (mean_value - last) * 0.35
        uncertainty = max(math.sqrt(variance) * 0.15, 5.0)
        results.append(float(rng.gauss(trend, uncertainty)))
        last = results[-1]

    return results


def _linear_forecast(values: Sequence[float]) -> float:
    """Forecast the next value using a simple linear trend."""

    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    if all(val == values[0] for val in values):
        return float(values[0])

    x_values = list(range(n))
    mean_x = sum(x_values) / n
    mean_y = sum(values) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, values))
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    slope = numerator / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    return slope * n + intercept


def forecast_disease_trends(
    disease_rows: List[Dict[str, object]],
    region_context: Optional[Dict[Tuple[str, str], RegionContext]] = None,
    horizon: int = 1,
) -> List[ForecastResult]:
    """Forecast near-term patient volumes for each region and condition."""

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in disease_rows:
        key = (row["country"], row["region"])
        grouped.setdefault(key, []).append(row)

    forecasts: List[ForecastResult] = []
    for (country, region), rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda item: item["month_dt"])
        latest = sorted_rows[-1]
        months = [row["month_dt"] for row in sorted_rows]
        context = region_context.get((country, region)) if region_context else None

        base_features, future_features = _build_time_features(months, context, horizon=horizon)
        for indicator in ("malaria_cases", "respiratory_cases", "maternal_complications"):
            history = [float(item[indicator]) for item in sorted_rows]
            current_value = float(latest[indicator])

            if len(history) < 3:
                forecast_value = _linear_forecast(history)
                ci_width = max(abs(forecast_value - current_value) * 0.15, 10.0)
                ci = (forecast_value - ci_width, forecast_value + ci_width)
            else:
                model = GradientBoostingTimeSeries()
                try:
                    model.fit(base_features, history)
                    gb_forecast = float(model.predict(future_features)[-1])
                except ValueError:
                    gb_forecast = _linear_forecast(history)

                structural = _structural_forecast(history, horizon=horizon)[-1]
                forecast_value = float(0.7 * gb_forecast + 0.3 * structural)
                ci_width = max(abs(forecast_value - current_value) * 0.25, 15.0)
                ci = (forecast_value - ci_width, forecast_value + ci_width)

            forecasts.append(
                ForecastResult(
                    country=country,
                    region=region,
                    indicator=indicator,
                    current_value=current_value,
                    forecast_value=forecast_value,
                    delta=forecast_value - current_value,
                    confidence_interval=ci,
                )
            )
    return forecasts


TRIAGE_GUIDELINES: Dict[str, str] = {
    "clinic": "Stabilise emergencies, manage malaria/ARI per IMCI, escalate obstetric emergencies.",
    "health_center": "Provide IMCI, basic emergency obstetric care, refer severe complications within 2 hours.",
    "hospital": "Stabilise all referrals, ensure blood products, and coordinate inter-facility transfers.",
}


def recommend_resources(
    facilities: Sequence[Dict[str, object]],
    surge_alerts: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[Dict[str, object]]:
    """Generate staffing, commodity, and triage recommendations per facility."""

    recommendations: List[Dict[str, object]] = []
    surge_alerts = surge_alerts or {}

    for facility in facilities:
        pop = facility["catchment_population"]
        current_staff = facility["staff_count"]
        target_staff = (18 * pop) / 10000
        required_staff = max(0, round(target_staff - current_staff))
        urgent_stock = facility["stockout_rate"] > 0.35
        referral_needed = (
            facility["functionality_score"] < 0.5 and facility["daily_visits"] > 120
        )
        surge_delta = surge_alerts.get((facility["country"], facility["region"]), 0.0)
        workload_per_staff = (
            facility["daily_visits"] / current_staff if current_staff else facility["daily_visits"]
        )
        overload = workload_per_staff > 35 or surge_delta > 40

        recommendations.append(
            {
                "country": facility["country"],
                "region": facility["region"],
                "facility_id": facility["facility_id"],
                "facility_type": facility["facility_type"],
                "staff_count": current_staff,
                "required_staff": required_staff,
                "urgent_stock_replenishment": urgent_stock,
                "refer_patients": referral_needed,
                "surge_alert": overload,
                "surge_driver": surge_delta,
                "triage_guideline": TRIAGE_GUIDELINES.get(
                    facility["facility_type"].lower(),
                    "Escalate life-threatening emergencies per national PHC protocols.",
                ),
            }
        )
    return recommendations


def compile_insights(language: str = DEFAULT_LANGUAGE) -> Dict[str, object]:
    """Generate a consolidated package of insights for downstream use."""

    facilities = compute_facility_gaps(load_facility_data())
    disease_rows = load_disease_data()
    region_context = _aggregate_region_context(facilities)

    country_view = aggregate_country_landscape(facilities)
    forecasts = forecast_disease_trends(disease_rows, region_context=region_context)
    surge_alerts: Dict[Tuple[str, str], float] = {}
    for forecast in forecasts:
        if forecast.indicator != "respiratory_cases":
            continue
        if forecast.delta > 30:
            surge_alerts[(forecast.country, forecast.region)] = forecast.delta

    recommendations = recommend_resources(facilities, surge_alerts=surge_alerts)

    underserved = [facility for facility in facilities if facility["undercapacity"]]
    underserved = sorted(underserved, key=lambda row: (row["country"], row["staff_per_10k"]))[:10]

    underserved_records = [
        {
            "country": row["country"],
            "region": row["region"],
            "facility_id": row["facility_id"],
            "staff_per_10k": row["staff_per_10k"],
            "functionality_score": row["functionality_score"],
            "stockout_rate": row["stockout_rate"],
        }
        for row in underserved
    ]

    base_payload = {
        "country_summary": country_view,
        "underserved_facilities": underserved_records,
        "forecasts": [forecast.as_dict() for forecast in forecasts],
        "recommendations": recommendations,
    }

    if localization:
        base_payload["localized_messages"] = localization.localise_insights(
            base_payload, language
        )

    return base_payload


def print_insights(insights: Dict[str, object], language: str = DEFAULT_LANGUAGE) -> None:
    """Render insights in a human-readable, localised format."""

    localized = insights.get("localized_messages") if localization else None

    if localized:
        print(localized.get("headline", "=== PHC Snapshot ==="))
    else:
        print("=== Country Landscape ===")

    for country in insights["country_summary"]:
        summary = (
            f"{country['country']}: {country['facilities']} facilities | "
            f"Avg staff/10k={country['avg_staff_per_10k']:.1f} | "
            f"Functionality={country['avg_functionality']:.2f} | "
            f"Stock-outs={country['stockout_rate']:.2f} | "
            f"Undercapacity rate={country['undercapacity_rate']:.2f}"
        )
        print(summary)

    print("\n=== Priority Facilities ===")
    if not insights["underserved_facilities"]:
        message = (
            localized.get("no_priority_facilities")
            if localized
            else "All facilities meet minimum standards."
        )
        print(message)
    else:
        for facility in insights["underserved_facilities"]:
            print(
                f"{facility['country']} - {facility['region']} - {facility['facility_id']}: "
                f"staff/10k={facility['staff_per_10k']:.1f}, "
                f"functionality={facility['functionality_score']:.2f}, "
                f"stock-out={facility['stockout_rate']:.2f}"
            )

    print("\n=== Forecast Alerts ===")
    for forecast in insights["forecasts"]:
        if abs(forecast["delta"]) < 20:
            continue
        direction = "increase" if forecast["delta"] > 0 else "decrease"
        ci_text = f"CI[{forecast['confidence_interval'][0]:.1f}, {forecast['confidence_interval'][1]:.1f}]"
        print(
            f"{forecast['country']} - {forecast['region']} ({forecast['indicator']}): "
            f"{direction} of {abs(forecast['delta']):.1f} expected next month | {ci_text}"
        )

    print("\n=== Resource Recommendations ===")
    for rec in insights["recommendations"]:
        actions: List[str] = []
        if rec["required_staff"] > 0:
            actions.append(f"deploy {int(rec['required_staff'])} staff")
        if rec["urgent_stock_replenishment"]:
            actions.append("expedite essential medicines")
        if rec["refer_patients"]:
            actions.append("activate referral support")
        if rec.get("surge_alert"):
            actions.append("prepare surge clinic and overflow triage")

        if actions:
            print(f"{rec['country']} - {rec['facility_id']}: " + ", ".join(actions))
            print(f"  Triage: {rec['triage_guideline']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PHC insights and recommendations.")
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path to export insights as JSON",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code for localised messaging (e.g. en, fr, sw)",
    )
    parser.add_argument(
        "--channel-dir",
        type=Path,
        help="Optional directory to export RapidPro/USSD/SMS assets",
    )

    args = parser.parse_args()

    insights = compile_insights(language=args.language)
    print_insights(insights, language=args.language)

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("w", encoding="utf-8") as handle:
            json.dump(insights, handle, indent=2)
        print(f"\nInsights exported to {args.export}")

    if args.channel_dir:
        channel_integrations.export_channel_assets(
            insights,
            args.channel_dir,
            language=args.language,
        )


if __name__ == "__main__":
    main()
