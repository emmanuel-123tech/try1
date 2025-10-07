"""Streamlit interface for low-bandwidth PHC decision support."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:  # pragma: no cover - handle package/script execution
    from .phc_analysis import compile_insights
    from . import localization
    from .channel_integrations import format_ussd_menu
except ImportError:  # pragma: no cover
    from phc_analysis import compile_insights  # type: ignore
    import localization  # type: ignore
    from channel_integrations import format_ussd_menu  # type: ignore

DATA_DIR = Path("data")


def _load_facility_table() -> pd.DataFrame:
    facilities_path = DATA_DIR / "phc_facility_status.csv"
    if facilities_path.exists():
        return pd.read_csv(facilities_path)
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="PHC Intelligence", layout="wide")
    st.title("Primary Health Care Intelligence Console")

    languages = localization.available_languages()
    selected_language = st.sidebar.selectbox("Language", languages, index=languages.index(localization.DEFAULT_LANGUAGE))

    insights = compile_insights(language=selected_language)
    localized = insights.get("localized_messages", {})

    st.subheader(localized.get("headline", "PHC Snapshot"))
    cols = st.columns(3)
    for idx, country in enumerate(insights["country_summary"]):
        with cols[idx % 3]:
            st.metric(
                country["country"],
                f"{country['avg_staff_per_10k']:.1f} staff/10k",
                delta=f"Stock-outs {country['stockout_rate']:.2f}",
            )

    st.markdown("### Priority Facilities")
    if insights["underserved_facilities"]:
        df_priority = pd.DataFrame(insights["underserved_facilities"]) \
            .rename(columns={"facility_id": "Facility", "staff_per_10k": "Staff per 10k"})
        st.dataframe(df_priority)
    else:
        st.info(localized.get("no_priority_facilities", "All facilities meet minimum standards."))

    st.markdown("### Forecast Alerts")
    alerts = [f for f in insights["forecasts"] if abs(f["delta"]) >= 20]
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        st.dataframe(df_alerts[["country", "region", "indicator", "forecast_value", "delta", "confidence_interval"]])
    else:
        st.success("No significant forecast alerts detected")

    st.markdown("### Actionable Recommendations")
    df_recommendations = pd.DataFrame(insights["recommendations"]) \
        .rename(columns={"required_staff": "Staff Gap", "urgent_stock_replenishment": "Restock"})
    st.dataframe(df_recommendations[["country", "facility_id", "Staff Gap", "Restock", "surge_alert", "triage_guideline"]])

    st.markdown("### Offline / USSD View")
    st.code(format_ussd_menu(insights, language=selected_language))

    facilities_df = _load_facility_table()
    if not facilities_df.empty:
        st.markdown("### Facility Readiness Explorer")
        filtered_country = st.selectbox("Filter by country", ["All"] + sorted(facilities_df["country"].unique().tolist()))
        df = facilities_df
        if filtered_country != "All":
            df = df[df["country"] == filtered_country]
        st.bar_chart(df.groupby("region")["functionality_score"].mean())

    st.caption("Streamlit app optimised for low-bandwidth deployments. Cache static datasets locally for offline use.")


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
