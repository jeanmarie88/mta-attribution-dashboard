# Multi-Touch Attribution - Digital Channel Credit Allocation

Heuristic and data-driven attribution across the digital customer journey

An interactive companion to a multi-touch attribution case study on an anonymised
digital touchpoint dataset (10,000 users, 52,133 touchpoints,
337 conversions over a 2023-01-02 to 2023-05-07 window).

Five attribution models are compared: first touch, last touch, time decay, Markov
removal effect and Shapley value.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push this folder to a public GitHub repository and point
[share.streamlit.io](https://share.streamlit.io) at `app.py`.

## Contents

- `app.py` - the dashboard
- `data/` - pre-aggregated CSVs (no user-level data)
