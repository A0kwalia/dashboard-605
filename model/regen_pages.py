"""Regenerate the model-driven dashboard pages on the KSI model.

Follows notebook cells B18.1-B18.4 (assignment1.ipynb): same page template and the same chart
engine (copied unchanged from the existing pages apart from its outcome wording), fed with the
KSI predictions for the held-out 2024 year.
Usage: python regen_pages.py <artefacts.pkl> <model_ksi.json> <dashboard_dir> <10_crash_scatter.html to take the chart engine from>
"""
import sys, json, re
import numpy as np
import pandas as pd
import pickle

art = pickle.load(open(sys.argv[1], "rb"))
meta = json.load(open(sys.argv[2]))["meta"]
D = sys.argv[3]
err, imp = art["err"], art["imp"]
AVG = meta["state_average"]
MAROON = "#8C1D40"

# ---- chart engine: lifted from the existing page, KSI wording ----
old = open(sys.argv[4], encoding="utf-8").read()
i = old.find("var SPEC="); j = old.find("</script>", i)
core = old[j + len("</script>"):]
core = core[core.find("<script>") + len("<script>"):core.rfind("</script>")]
if '"fatal or serious injury (KSI)"' not in core:   # first run on the original casualty page
    assert '((+yy[best])?"casualty":"non-casualty")' in core
    core = core.replace('((+yy[best])?"casualty":"non-casualty")',
                        '((+yy[best])?"fatal or serious injury (KSI)":"not KSI")')

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{margin:0;padding:26px;font-family:Calibri,Segoe UI,system-ui,sans-serif;color:#21252B;background:#fff;max-width:900px}}
h1{{font-family:Cambria,Georgia,serif;font-size:21px;margin:0 0 4px}}
.sub{{color:#5A5F66;font-size:13.5px;line-height:1.55;margin:0 0 16px;max-width:760px}}
.ctl{{margin:2px 0 10px;font-size:13px;color:#5A5F66}}
input[type=range]{{width:340px;vertical-align:middle}}
svg{{max-width:100%;height:auto;display:block;border:1px solid #EDEAEB;border-radius:6px}}
.cap{{font-size:12px;color:#5A5F66;font-style:italic;margin:10px 0 0;max-width:760px;line-height:1.55}}
.src{{margin-top:22px;padding-top:12px;border-top:1px solid #DDD7D9;font-size:11.5px;color:#5A5F66}}
.tip{{position:fixed;pointer-events:none;background:#21252B;color:#fff;font-size:12px;padding:7px 10px;
border-radius:5px;opacity:0;transition:opacity .12s;z-index:99;line-height:1.5;max-width:240px}}
</style></head><body>
<h1>{title}</h1>
<p class="sub">{sub}</p>
{controls}
<svg id="c"></svg>
<canvas id="cv" style="display:none;max-width:100%;border:1px solid #EDEAEB;border-radius:6px"></canvas>
<p class="cap">{cap}</p>
<div class="src">{src}</div>
<div class="tip" id="tip"></div>
<script>var SPEC={spec};</script>
<script>{core}</script>
</body></html>"""

MAP_CONTROLS = """<div class="ctl">Highlight areas with mean predicted KSI risk above <b id="thv"></b>
&nbsp;&nbsp;<span id="count"></span><br><input type="range" id="slider"></div>"""
SCATTER_CONTROLS = """<div class="ctl">
<label>Show crashes with predicted KSI risk above <b id="thv"></b></label><br>
<input type="range" id="slider">
&nbsp;&nbsp;<label>Actual outcome</label>
<select id="fout"><option value="a">All</option><option value="1">KSI crashes only</option>
<option value="0">Other crashes only</option></select>
&nbsp;&nbsp;<label>Lighting</label><select id="flight"><option value="a">All</option></select>
<br><span id="count"></span></div>"""
SRC = ("Source: KSI model (histogram gradient boosting, sigmoid calibration), trained on 2020&ndash;2023 and "
       "evaluated on the held-out 2024 year &middot; NSW Road Crash Data 2020&ndash;2024, Transport for NSW "
       "(CC BY 4.0) &middot; Hover any element for exact values.")

def save(name, title, sub, cap, spec, controls=""):
    html = PAGE.format(title=title, sub=sub, cap=cap, controls=controls,
                       spec=json.dumps(spec), core=core, src=SRC)
    open(f"{D}/{name}", "w", encoding="utf-8").write(html)
    print(name, len(html))

pct = lambda v: f"{v*100:.1f}%"

# 06 risk bands
bands = meta["bands"]
items = [{"l": b["label"], "l2": "n = " + format(b["n"], ","), "v": b["predicted"], "v2": b["observed"],
          "note": "%d crashes in 2024" % b["n"]} for b in bands]
lims = [f'{b["label"]} {"under " + pct(bands[1]["from"]) if k == 0 else ("from " + pct(b["from"]))}'
        for k, b in enumerate(bands)]
hi, el = bands[-1], bands[-2]
save("06_risk_band_validation.html", "Do the risk bands mean what they say?",
     "Mean predicted KSI risk against the KSI rate actually observed within each band, on the 2024 test year. "
     "Bands: " + ", ".join(lims) + ".",
     f"The bands rank correctly: each band's observed KSI rate is higher than the one below it. The upper bands "
     f"understate risk in 2024, though: crashes rated High were KSI {pct(hi['observed'])} of the time against a "
     f"predicted {pct(hi['predicted'])}. 2024 had more KSI crashes than the 2020&ndash;2023 training years "
     f"({pct(float(err.y_true.mean()))} against {pct(AVG)}), so read a band as a ranking and quote the observed "
     f"rate, not the predicted one, when describing it. High holds {hi['n']:,} of {len(err):,} crashes.",
     {"type": "bar", "w": 640, "h": 420, "max": round(min(1.0, max(b["observed"] for b in bands) * 1.15 + 0.02), 2),
      "bottom": 60, "lab1": "Mean predicted", "lab2": "Observed KSI rate", "ylabel": "KSI rate", "items": items})

# 07 LGA map
lga = (err.groupby("LGA").agg(x=("Longitude", "median"), y=("Latitude", "median"), v=("p_pred", "mean"),
                              obs=("y_true", "mean"), n=("y_true", "size")).query("n >= 60"))
lga = lga[(lga.y.between(-38, -27)) & (lga.x.between(140, 154))]
pts = [{"l": k, "x": float(r.x), "y": float(r.y), "v": float(r.v), "obs": float(r.obs), "n": int(r.n)}
       for k, r in lga.iterrows()]
metro = lga[lga.x > 150.5].v.mean(); inland = lga[lga.x <= 150.5].v.mean()
save("07_lga_risk_map.html", "Risk map by local government area",
     "Mean predicted KSI risk per LGA across the 2024 test year, for LGAs with at least 60 crashes. Bubble size "
     "is crash volume, colour is predicted risk. Drag the slider to highlight LGAs above a chosen threshold.",
     f"Mean predicted KSI risk ranges from {pct(lga.v.min())} ({lga.v.idxmin()}) to {pct(lga.v.max())} "
     f"({lga.v.idxmax()}). Inland and regional LGAs (west of 150.5&deg;E) average {pct(inland)} against "
     f"{pct(metro)} for coastal and metropolitan LGAs. The map summarises the model's output by area; it does "
     f"not mean location itself causes the risk.",
     {"type": "map", "w": 660, "h": 540, "points": pts, "thresh": AVG}, controls=MAP_CONTROLS)

# 08 LGA ranking
t = (err.groupby("LGA").agg(pred=("p_pred", "mean"), obs=("y_true", "mean"), n=("y_true", "size"))
     .query("n >= 150").sort_values("pred", ascending=False).head(15))
lp = art["lga_perf"].set_index("level")["ROC_AUC"]
rank_items = [{"l": k, "v": float(r.pred), "color": MAROON,
               "note": "observed %.3f &middot; %d crashes &middot; gap %+.3f%s" % (
                   r.obs, int(r.n), r.pred - r.obs,
                   " &middot; low model reliability here" if lp.get(k, 1) < 0.60 else "")}
              for k, r in t.iloc[::-1].iterrows()]
gap = (t.pred - t.obs)
lo_v = np.floor((t.pred.min() - 0.02) * 50) / 50; hi_v = np.ceil((t.pred.max() + 0.01) * 50) / 50
save("08_lga_ranking.html", "Highest predicted-risk LGAs",
     "The fifteen LGAs with the highest mean predicted KSI risk, among those with at least 150 crashes in 2024. "
     "Hover for the observed KSI rate and the prediction gap.",
     f"{t.index[0]} ranks first at {pct(t.pred.iloc[0])}. {int((gap < 0).sum())} of these 15 LGAs had a higher "
     f"observed KSI rate than predicted, by up to {abs(gap.min())*100:.1f} points ({gap.idxmin()}), consistent "
     f"with 2024's higher statewide KSI rate. Use the order, not the exact values. Areas marked \"low model "
     f"reliability\" had a 2024 ROC-AUC below 0.60.",
     {"type": "hbar", "w": 660, "h": 470, "left": 195, "min": float(max(0, lo_v)), "max": float(hi_v), "ref": AVG,
      "refLabel": "state average", "lab1": "Predicted KSI risk", "xlabel": "Mean predicted KSI risk",
      "items": rank_items})

# 09 permutation importance
top = imp.head(12).iloc[::-1]
nice = {"SPEED_LIMIT_KMH": "Speed limit"}
items = [{"l": nice.get(r["feature"], r["feature"]), "v": float(r["auc_drop"]), "color": MAROON,
          "note": "std %.4f" % r["std"]} for _, r in top.iterrows()]
first = imp.iloc[0]; geo = imp[imp.feature.isin(["LGA", "Latitude", "Longitude", "Conurbation 1"])].auc_drop.sum()
save("09_feature_importance.html", "What the model relies on",
     "Drop in ROC-AUC when each feature is randomly shuffled, measured on a held-out 2024 sample so it reflects "
     "generalisation rather than training fit. The twelve most important of 24 features are shown.",
     f"{nice.get(first.feature, first.feature)} and speed limit carry the most signal. Unlike the earlier "
     f"casualty model, location now matters: LGA, latitude, longitude and conurbation together account for an "
     f"ROC-AUC drop of {geo:.3f}. Bar length shows what the model uses, not how much risk a condition causes; "
     f"the Scenario explorer shows the direction of each factor for a specific site.",
     {"type": "hbar", "w": 660, "h": 460, "left": 200, "vfmt": "4", "lab1": "ROC-AUC drop",
      "xlabel": "Drop in ROC-AUC when the feature is shuffled", "items": items})

# 10 crash scatter (as notebook B18.4)
g = err.dropna(subset=["Latitude", "Longitude"]).copy()
g = g[(g.Latitude.between(-38, -27)) & (g.Longitude.between(140, 154))]
lga_names = sorted(g["LGA"].unique().tolist())
li_names = ["Daylight", "Dusk", "Dawn", "Darkness", "Unknown"]
wx_names = sorted(g["Weather"].astype(str).unique().tolist())
sp_names = sorted(g["Speed limit"].astype(str).unique().tolist(),
                  key=lambda s: int(s.split()[0]) if s.split()[0].isdigit() else 999)
li_i = {v: k for k, v in enumerate(li_names)}; wx_i = {v: k for k, v in enumerate(wx_names)}
sp_i = {v: k for k, v in enumerate(sp_names)}; lga_i = {v: k for k, v in enumerate(lga_names)}
assert len(li_names) < 10 and len(wx_names) < 10
spec = {"type": "scatter", "w": 680, "h": 560,
        "lon": [round(float(v), 3) for v in g["Longitude"]], "lat": [round(float(v), 3) for v in g["Latitude"]],
        "p": [round(float(v), 3) for v in g["p_pred"]], "y": "".join(str(int(v)) for v in g["y_true"]),
        "li": "".join(str(li_i.get(str(v), 4)) for v in g["Natural lighting"]),
        "wx": "".join(str(wx_i.get(str(v), 0)) for v in g["Weather"]),
        "sp": [sp_i.get(str(v), 0) for v in g["Speed limit"]], "lga": [lga_i[v] for v in g["LGA"]],
        "lgaNames": lga_names, "liNames": li_names, "wxNames": wx_names, "spNames": sp_names,
        "vmin": float(round(g.p_pred.min(), 3)), "vmax": float(round(g.p_pred.max(), 3)), "xlabel": "Longitude"}
save("10_crash_scatter.html", "Predicted risk, individual crashes",
     "Every reported crash in the 2024 test year, positioned by coordinates and coloured by the KSI risk the model "
     "predicted for it. Filter by predicted risk, actual outcome and lighting; hover any point for its conditions "
     "and what actually happened.",
     f"Compare the two outcome filters. KSI and other crashes overlap heavily in space and colour, which is the "
     f"spatial expression of a ROC-AUC of {meta['test_roc_auc']:.3f}: real separation, but modest. Raise the "
     f"slider to {bands[-1]['from']:.2f} to see the High band, where {pct(bands[-1]['observed'])} of crashes "
     f"were KSI.",
     spec, controls=SCATTER_CONTROLS)

# review page: replace its ITEMS with the new ranking
fb = open(f"{D}/user_feedback.html", encoding="utf-8").read()
new_items = "var ITEMS = [\n" + ",\n".join(
    '  {l:%s, v:%.4f, note:%s}' % (json.dumps(k), r.pred, json.dumps(
        "observed %.3f · %d crashes · gap %+.3f" % (r.obs, int(r.n), r.pred - r.obs)))
    for k, r in t.iloc[::-1].iterrows()) + "\n];"
fb = re.sub(r"var ITEMS = \[.*?\];", lambda m: new_items, fb, flags=re.S)
fb = fb.replace("each of the fifteen highest predicted-risk council areas",
                "each of the fifteen highest predicted KSI-risk council areas")
fb = fb.replace("<th>Predicted risk</th>", "<th>Predicted KSI risk</th>")
fb = fb.replace("Source: assignment1_updated.ipynb, Part B &middot; NSW Road Crash Data",
                "Source: KSI model, held-out 2024 year &middot; NSW Road Crash Data")
open(f"{D}/user_feedback.html", "w", encoding="utf-8").write(fb)
print("user_feedback.html updated")
