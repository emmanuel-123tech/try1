# AI for Smarter Primary Health Care in Africa

This repository contains a lightweight, end-to-end prototype that demonstrates how artificial intelligence and data analytics can strengthen Primary Health Care (PHC) systems across African countries. The project combines open data framing, predictive analytics, and decision-support tooling ideas into an integrated package suitable for hackathon delivery. The refreshed release adds automated data enrichment, gradient-boosted forecasting with structural priors, localisation, and low-bandwidth user interfaces.

## Solution Overview

The solution focuses on four complementary capabilities:

1. **Comprehensive PHC Data Landscape** – Harmonises facility readiness, service utilisation, and disease incidence indicators to reveal underserved populations and infrastructure gaps.
2. **Predictive Insights & Diagnostic Support** – Applies an embedded gradient boosting time-series learner blended with Bayesian structural priors to anticipate patient surges and stock-out risks while surfacing early warning signals for common PHC conditions.
3. **Decision-Support Prototype** – Generates actionable recommendations for staffing, drug inventory, triage protocols, and surge planning that can be embedded into dashboards, USSD menus, or RapidPro chatbots.
4. **Scalability & Impact Plan** – Localises the output into English, French, Swahili, and Portuguese with offline caches and an extendable channel toolkit ready for deployment across diverse African contexts.

## Repository Structure

```
├── data/
│   ├── cache/                         # API cache artefacts for offline use
│   ├── phc_facility_status.csv        # Sample facility readiness and utilisation indicators
│   └── disease_incidence.csv          # Monthly primary-care case trends by region
├── src/
│   ├── phc_analysis.py                # Core analytics pipeline and recommendation engine
│   ├── data_refresh.py                # Connectors for World Bank, WHO, DHIS2, and OpenHIE APIs
│   ├── localization.py                # Local language packaging utilities
│   ├── channel_integrations.py        # RapidPro / USSD export helpers
│   └── streamlit_app.py               # Low-bandwidth decision-support UI
├── reports/
│   ├── generated_insights.json        # Auto-generated insight bundle (created at runtime)
│   └── solution_overview.md           # Narrative on vision, adoption strategy, and impact pathways
└── README.md
```

## Getting Started

1. (Optional) create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analytics pipeline to generate summary insights and recommendations:
   ```bash
   python src/phc_analysis.py --language fr --export reports/generated_insights.json
   ```
   The `--language` flag controls localisation for console output, JSON payloads, and generated channel assets.
4. (Optional) Refresh datasets from open APIs:
   ```bash
   python src/data_refresh.py --country KEN --indicator SH.MED.PHYS.ZS
   ```
5. (Optional) Launch the Streamlit decision-support workspace:
   ```bash
   streamlit run src/streamlit_app.py
   ```

The analytics script prints country- and region-level insights to the console and optionally exports them as machine-readable JSON for dashboards, RapidPro flows, or messaging bots. The extended implementation remains dependency-light while bundling numpy, pandas, and Streamlit for richer modelling and UI options.

## Data Inputs

The repository ships with illustrative sample datasets constructed from publicly available statistics such as the World Bank World Development Indicators, WHO Service Availability and Readiness Assessments, and national DHIS2 portals. Replace these placeholders with live feeds or API connectors using `src/data_refresh.py`.

- `phc_facility_status.csv` includes staffing ratios, patient throughput, stock-out rates, and functionality scores for facilities across Nigeria, Kenya, Uganda, and South Africa.
- `disease_incidence.csv` tracks monthly cases for malaria, respiratory infections, and maternal health complications to support surge prediction.
- `data/cache/` stores cached API responses to ensure offline continuity for low-connectivity deployments.

## Extending the Prototype

- **Data enrichment**: Use `OpenDataRefresher` in `src/data_refresh.py` to sync new indicators from World Bank, WHO, DHIS2 analytics endpoints, or OpenHIE FHIR bundles. Cache artefacts under `data/cache/` for offline resilience.
- **Advanced modelling**: Tune the gradient boosting ensemble, extend the structural prior, or plug in neural sequence models to capture seasonality, interventions, and climatic drivers.
- **User interfaces**: Pair the Streamlit console with Power BI or embed JSON outputs in RapidPro, USSD/SMS gateways, and WhatsApp bots using `src/channel_integrations.py`.
- **Localisation**: Expand `src/localization.py` with additional languages, adapt triage guidance to national clinical protocols, and pre-generate SMS/USSD assets for low-literacy or offline workflows.

## License

This project is released under the MIT License. Feel free to adapt and extend it for hackathon or production deployments.
