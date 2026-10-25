"""Export the calibrated KSI model to a compact JSON file the dashboard evaluates in JavaScript.

The calibrated model is CalibratedClassifierCV(cv=3, sigmoid): three (pipeline, sigmoid) pairs whose
calibrated probabilities are averaged. Each pipeline is OneHotEncoder(min_frequency=30) +
median-impute/standard-scale -> HistGradientBoosting. The export folds the preprocessing into the
tree splits, so the browser works on raw inputs:
  * a split on a one-hot column becomes "category code in set -> go right";
  * a split on a scaled numeric column becomes "raw value <= threshold -> go left", with the
    fold's training median substituted for a missing value.
Usage: python export_model.py <artefacts.pkl> <out_dir>
"""
import sys, json, pickle
import numpy as np
import pandas as pd

art = pickle.load(open(sys.argv[1], "rb"))
OUT = sys.argv[2]
cal = art["calibrated"]
CATS, NUMS, FEATS = art["CATEGORICAL_FEATURES"], art["NUMERIC_FEATURES"], art["MODEL_FEATURES"]
adf = art["analysis_df"]
dev = adf[adf["Year of crash"] <= 2023]

# global vocabulary per categorical feature, most frequent first
vocab = {c: adf[c].value_counts().index.astype(str).tolist() for c in CATS}
code = {c: {v: i for i, v in enumerate(vocab[c])} for c in CATS}

def export_fold(cc):
    pipe = cc.estimator
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    assert type(clf).__name__ == "HistGradientBoostingClassifier", type(clf).__name__
    ohe = pre.named_transformers_["cat"]
    numpipe = pre.named_transformers_["num"]
    imp, sc = numpipe.named_steps["imp"], numpipe.named_steps["sc"]

    # map every (feature, vocab category) to its output column, by transforming probes
    n_out = ohe._n_features_outs
    starts = np.concatenate([[0], np.cumsum(n_out)])
    col_owner = {}          # output column -> (feature index in CATS, set of vocab codes hot there)
    for j, c in enumerate(CATS):
        probe = pd.DataFrame({k: [vocab[k][0]] * len(vocab[c]) for k in CATS})
        probe[c] = vocab[c]
        Z = ohe.transform(probe)[:, starts[j]:starts[j + 1]]
        for col in range(n_out[j]):
            col_owner[starts[j] + col] = (j, set(np.where(Z[:, col] == 1)[0].tolist()))
    n_cat_cols = int(starts[-1])

    sets, trees = [], []
    for it in clf._predictors:
        nodes = it[0].nodes
        tree = []
        for nd in nodes:
            if nd["is_leaf"]:
                tree.append([-1, round(float(nd["value"]), 7)])
                continue
            fi, thr = int(nd["feature_idx"]), float(nd["num_threshold"])
            assert not nd["is_categorical"]
            if fi < n_cat_cols:
                j, hot = col_owner[fi]
                # value is 1 for codes in `hot`, else 0; left iff value <= thr
                right = sorted(k for k in range(len(vocab[CATS[j]]))
                               if not ((1.0 if k in hot else 0.0) <= thr))
                sets.append(right)
                tree.append([FEATS.index(CATS[j]), "s", len(sets) - 1, int(nd["left"]), int(nd["right"])])
            else:
                k = fi - n_cat_cols
                raw_thr = thr * float(sc.scale_[k]) + float(sc.mean_[k])
                tree.append([FEATS.index(NUMS[k]), "n", round(raw_thr, 6), int(nd["left"]), int(nd["right"])])
        trees.append(tree)
    cal1 = cc.calibrators[0]   # _SigmoidCalibration: p = 1 / (1 + exp(a * score + b))
    return {"base": float(np.ravel(clf._baseline_prediction)[0]),
            "median": [float(v) for v in imp.statistics_],
            "trees": trees, "sets": sets,
            "sig_a": float(cal1.a_), "sig_b": float(cal1.b_)}

folds = [export_fold(cc) for cc in cal.calibrated_classifiers_]

# per-LGA defaults: median coordinates, modal conurbation and urbanisation
lga_defaults = {}
for lga, g in adf.groupby("LGA"):
    lga_defaults[lga] = {"Latitude": round(float(g.Latitude.median()), 4) if g.Latitude.notna().any() else None,
                         "Longitude": round(float(g.Longitude.median()), 4) if g.Longitude.notna().any() else None,
                         "Conurbation 1": str(g["Conurbation 1"].mode().iloc[0]),
                         "Urbanisation": str(g["Urbanisation"].mode().iloc[0]),
                         "n": int(len(g))}

mode_row = {c: str(adf[c].mode().iloc[0]) for c in CATS}
mode_row["SPEED_LIMIT_KMH"] = float(adf["SPEED_LIMIT_KMH"].median())

# background sample of development-period crashes for the factor explanations
bg = dev.sample(n=150, random_state=7)
background = [[code[c][str(r[c])] if c in CATS else (None if pd.isna(r[c]) else round(float(r[c]), 4))
               for c in FEATS] for _, r in bg.iterrows()]

# reliability per LGA on the 2024 test year
lga_perf = {r["level"]: {"auc": round(float(r["ROC_AUC"]), 3), "n": int(r["n"])}
            for _, r in art["lga_perf"].iterrows()}

# risk bands on the KSI probability scale; observed rates from the held-out 2024 year
err = art["err"]
BAND_DEF = [(0.0, "Low", "#6B7178"), (0.20, "Moderate", "#B26A84"),
            (0.25, "Elevated", "#9C3A5C"), (0.35, "High", "#6E1633")]
bands = []
for i, (lo, label, color) in enumerate(BAND_DEF):
    hi = BAND_DEF[i + 1][0] if i + 1 < len(BAND_DEF) else 9
    g = err[(err.p_pred >= lo) & (err.p_pred < hi)]
    bands.append({"from": lo, "label": label, "color": color, "n": int(len(g)),
                  "predicted": round(float(g.p_pred.mean()), 4), "observed": round(float(g.y_true.mean()), 4)})
print(pd.DataFrame(bands))

s = art["summary"]
model = {
    "meta": {"target": "KSI (fatal or serious injury)", "model": s["selected_model"] + " + sigmoid calibration",
             "trained_on": "2020-2023", "test_year": 2024, "test_roc_auc": s["test_roc_auc"],
             "roc_auc_95ci": s["roc_auc_95ci"], "ece": s["expected_calibration_error"],
             "state_average": round(float(dev["KSI"].mean()), 4), "bands": bands},
    "features": FEATS, "categorical": CATS, "numeric": NUMS,
    "vocab": vocab, "mode": mode_row, "lga_defaults": lga_defaults,
    "background": background, "lga_perf": lga_perf, "folds": folds,
}
path = f"{OUT}/model_ksi.json"
with open(path, "w") as f:
    json.dump(model, f, separators=(",", ":"))
# the same data as a script, so the page also works when opened straight from disk (file://)
with open(f"{OUT}/model_ksi.js", "w") as f:
    f.write("/* Generated by export_model.py - calibrated KSI model for the scenario explorer. */\n")
    f.write("window.MODEL_KSI=" + json.dumps(model, separators=(",", ":")) + ";\n")
print("wrote", path, round(len(open(path).read()) / 1e6, 2), "MB;",
      [len(fd["trees"]) for fd in folds], "trees per fold")

# parity fixture: python probabilities for 2000 test rows, as raw inputs
te = adf[adf["Year of crash"] == 2024].sample(n=2000, random_state=1)
p = cal.predict_proba(te[FEATS])[:, 1]
rows = [[code[c].get(str(r[c]), -1) if c in CATS else (None if pd.isna(r[c]) else float(r[c])) for c in FEATS]
        for _, r in te.iterrows()]
json.dump({"rows": rows, "p": [float(v) for v in p]}, open(f"{OUT}/parity_fixture.json", "w"))
print("parity fixture written")
