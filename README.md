# Predictive Road Safety Risk Mapping: dashboard

ICT605 Group Project (Online Flexible 1). Live site: https://a0kwalia.github.io/dashboard-605/

A static dashboard that ranks NSW road crash scenarios by the probability of a **KSI** outcome (killed or
seriously injured). Everything is plain HTML and JavaScript, so it runs on GitHub Pages with no server, API
key or database. The scenario explorer runs the trained model in the browser.

## Pages

| File | What it shows |
| --- | --- |
| `index.html` | Dashboard shell with tab navigation |
| `severity_map.html` | Observed crash outcomes (needs the Plotly CDN) |
| `10_crash_scatter.html` | 2024 crashes coloured by predicted KSI risk |
| `07_lga_risk_map.html` | Mean predicted KSI risk per LGA |
| `08_lga_ranking.html` | 15 highest predicted-risk LGAs |
| `scenario_explorer.html` | Set site and conditions, get KSI risk, band, reliability note and top-5 factors |
| `interactive_risk_factors.html` | Crash counts by weather, speed, lighting, urbanisation (needs the Plotly CDN) |
| `09_feature_importance.html` | Permutation importance |
| `06_risk_band_validation.html` | Predicted vs observed KSI rate per risk band |
| `user_feedback.html` | Accept or reject the model's call on the 15 highest-risk LGAs |
| `ksi_model.js`, `model_ksi.js` | Browser model engine and the exported model (~0.4 MB) |

## Model (2024 held-out test year)

Histogram gradient boosting with sigmoid calibration, trained on 2020&ndash;2023, 24 planning-time features
(allow-list as in the Group Assignment).

| Measure | Value |
| --- | --- |
| ROC-AUC (95% bootstrap CI) | 0.630 (0.621&ndash;0.638) |
| Strongest baseline, B2 LGA &times; speed &times; lighting rate | 0.570 |
| PR-AUC (prevalence 0.251) | 0.385 |
| Expected calibration error | 0.034 |
| Precision in top 5% / top 10% | 0.544 / 0.473 (lift 2.17 / 1.88) |
| Grouped CV, unseen LGAs / rolling-origin CV | 0.616 / 0.621 |

Full tables are in `model/outputs/`.

### Changes from the Group Assignment proposal

- **Target is KSI**, not SERIOUS_CRASH, as the proposal recommended.
- **Model selection:** on 2023 validation the three candidates finished within 0.005 ROC-AUC of each other
  (random forest 0.6225, gradient boosting 0.6193). Ties within 0.005 go to the model that can run in the
  browser. Gradient boosting was also best on 2024 (`test_results_all_candidates.csv`).
- **Calibration is sigmoid, not isotonic.** ROC-AUC and ECE are the same on 2024, but isotonic gave many
  different scenarios an identical probability, so the explorer could not show a change in conditions.
- **Risk bands are re-cut for KSI:** Low &lt; 0.20, Moderate 0.20&ndash;0.25, Elevated 0.25&ndash;0.35, High &ge; 0.35.

### Known limitations

- **2024 had more KSI crashes than the training years** (25.1% vs 22.2%), so the upper bands understate
  2024 risk (High: predicted 0.39, observed 0.58). The pages tell planners to read bands as a ranking and to
  quote the observed rate.
- **Darkness still shows a lower KSI rate per crash than daylight** (21.3% vs 23.8%, 2020&ndash;2024). KSI
  reduces the SERIOUS_CRASH inversion but does not remove it; the model now scores night and day about equally.
- **Some LGAs have weak local performance.** The explorer warns for LGAs with a 2024 ROC-AUC below 0.60 and
  for LGAs with too few crashes to check.

## Reproducing

Requires Python 3.9+ with pandas, numpy, scikit-learn (built with 1.9.1) and openpyxl, plus Node 18+ for the
parity check. Put `nsw_road_crash_data_2020-2024_crash.xlsx` in a working folder, then:

```bash
python -c "import pandas as pd; pd.read_excel('nsw_road_crash_data_2020-2024_crash.xlsx').to_pickle('raw.pkl')"
python model/train_ksi.py raw.pkl ai_ksi                 # ~3 min; metrics + ai_ksi/artefacts.pkl
python model/export_model.py ai_ksi/artefacts.pkl export # model_ksi.json / .js + parity fixture
node model/parity.js ./ksi_model.js export/model_ksi.json export/parity_fixture.json
cp export/model_ksi.js .
python model/regen_pages.py ai_ksi/artefacts.pkl export/model_ksi.json . 10_crash_scatter.html
```

The parity check scores 2,000 test crashes in both scikit-learn and the browser engine; the largest
difference is below 1e-7.

### How the top-5 factors are calculated

For each factor (location features are grouped), the explorer replaces that factor's value with the value
from each of 150 randomly sampled 2020&ndash;2023 crashes and averages the predictions. The factor's
contribution is the scenario's risk minus that average: how far this value moves the risk compared with the
conditions of a typical reported crash. These are associations in reported crashes, not causes.

Data: NSW Road Crash Data 2020&ndash;2024, Transport for NSW, CC BY 4.0.
