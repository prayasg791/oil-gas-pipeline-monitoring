# 🛢️ Oil & Gas Pipeline Incident Monitoring & Cost Estimator (2010–2026)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oil-gas-pipeline-monitoring-3utwm52hfbkycskznrzswb.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, full-stack data analytics and predictive machine learning dashboard monitoring over 15+ years of official US **PHMSA (Pipeline and Hazardous Materials Safety Administration)** incident records (2010–2026).

🔗 **Live Deployment:** [Launch Web Application](https://oil-gas-pipeline-monitoring-3utwm52hfbkycskznrzswb.streamlit.app/)

---

## 📌 Project Overview

Pipeline infrastructure is critical to energy transportation, yet equipment failure, corrosion, and excavation damage present massive environmental and financial risks. 

This platform aggregates historical accident records across the United States to provide real-time KPIs, multi-dimensional safety trends, interactive geospatial incident mapping, and an integrated **Random Forest Regression model** for dynamic incident cost forecasting.

---

## ✨ Key Features

- **Dynamic KPI Summaries:** Instant metrics tracking total incidents, cumulative financial damage ($M), and total net loss/volume release.
- **Geospatial Mapping:** Interactive US map visualizing incident clusters, pinpointing locations by commodity type, operator, and coordinates.
- **Root Cause & Commodity Breakdown:** Interactive visual breakdowns of primary failure categories (corrosion, equipment defects, excavation) and commodity distributions.
- **Predictive Analytics (ML Cost Estimator):** Scikit-Learn `RandomForestRegressor` model that predicts estimated total incident remediation costs based on user-simulated release volume inputs.
- **Temporal & Schema Flexibility:** Automatically adapts to multi-year PHMSA dataset schemas spanning from 2010 through 2026.

---

## 🛠️ Tech Stack & Architecture

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation & Analysis:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations:** [Plotly Express](https://plotly.com/python/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`RandomForestRegressor`, `train_test_split`)
- **Data Source:** [PHMSA Pipeline Safety Flagged Files](https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files)

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/prayasg791/oil-gas-pipeline-monitoring.git](https://github.com/prayasg791/oil-gas-pipeline-monitoring.git)
   cd oil-gas-pipeline-monitoring
