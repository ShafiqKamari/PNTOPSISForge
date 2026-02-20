# PNTOPSISForge v1.0

PNTOPSISForge is a Streamlit-based Decision Support System implementing Pythagorean Neutrosophic TOPSIS (PNTOPSIS).

## Key features
- Crisp decision matrix input (Excel/CSV), mapped to PNS triplets using 5, 7, 9, or 11-point linguistic scales
- Benefit/Cost criteria support (auto-detect top B/C row; editable in UI)
- Criteria weights: Equal or Manual (no auto-normalization; warning if sum of weights is not 1)
- Distance toggle: PN-Euclidean, PN-Hamming, FIQ (FIQ follows Eq. 23-24 with per-criterion exponent)
- Sensitivity analysis toggle to compare rankings under Euclidean vs Hamming vs FIQ side-by-side
- Computational time display (preprocessing and ranking)
- Export: Excel workbook (all intermediate matrices) and PDF report

## Input file format
Recommended (Excel):

Row 1 contains criterion types (B or C).
Row 2 onward contains alternatives and crisp scores.

| Alt | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
|   | B | B | C | C |
| A1 | 7 | 8 | 4 | 6 |

If the B/C row is missing, the app defaults all criteria to B and you can adjust before running.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Forge Suite vision
- COREXForge (Weighting Engine)
- PNTOPSISForge (Ranking Engine)
- PNVIKORForge (Compromise Engine)
