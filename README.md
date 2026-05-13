# Reask Parametric TC Pricer

A Streamlit app for pricing parametric tropical cyclone insurance policies using the Reask DeepCyc and Metryc APIs.

## Setup

Requires Python 3.9+.

```bash
cd reask_parametric_pricer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
source venv/bin/activate
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Enter your Reask API credentials in the sidebar
2. Upload a GeoJSON file of policy shapes (or define circles/WKT manually)
3. Define or upload a payout structure CSV (`shape_id, from, to, payout_percentage`)
4. Run the stochastic analysis (DeepCyc) and optionally the historic analysis (Metryc)
5. Review pricing in the Pricing tab and download the full Excel export

## Sample inputs

- `example_shapes.geojson` + `example_payout_structures.csv` — two independent shapes
- `example_shapes_concentric.geojson` + `example_payout_structures_concentric.csv` — concentric zones

## Notes

- Do not commit real client shape files or payout structures to this repository
- Do not commit Reask API credentials — enter them in the app sidebar at runtime
- The `venv/` directory is gitignored; each developer creates their own locally
