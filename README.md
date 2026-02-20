
# PNTOPSISForge v1.0

PNTOPSISForge is a Streamlit-based Decision Support System implementing
Pythagorean Neutrosophic TOPSIS (PNTOPSIS).

## Features
- 5, 7, 9, 11-point PNS linguistic scales
- Benefit/Cost handling
- Equal or Manual weights
- Distance toggle: PN-Euclidean | PN-Hamming | FIQ
- Sensitivity analysis (compare distances side-by-side)
- Computational time display
- Relative closeness ranking

## Forge Suite Vision
- COREXForge (Weighting Engine)
- PNTOPSISForge (Ranking Engine)
- PNVIKORForge (Compromise Engine)

## Run Locally
pip install -r requirements.txt
streamlit run app.py
