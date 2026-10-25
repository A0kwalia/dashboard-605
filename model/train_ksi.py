"""Retrain the Group Assignment pipeline (assignment1.ipynb, Part B) on the KSI target.

Mirrors notebook cells B1-B13 with TARGET = "KSI". Differences from the notebook:
  * TARGET is KSI (Fatal or Serious Injury) instead of SERIOUS_CRASH.
  * Calibration is sigmoid (Platt) rather than isotonic; see the note at the calibration step.
  * Baseline B3 uses rates estimated on development data rather than the
    hard-coded 0.74 / 0.66, which were SERIOUS_CRASH rates.
Usage: python train_ksi.py <raw.pkl> <out_dir>
"""
import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
raw_path, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

df = pd.read_pickle(raw_path)
analysis_df = df[df["Year of crash"].between(2020, 2024)].copy()
analysis_df["SERIOUS_CRASH"] = analysis_df["Degree of crash"].isin(["Fatal", "Injury"]).astype(int)
analysis_df["KSI"] = analysis_df["Degree of crash - detailed"].isin(["Fatal", "Serious Injury"]).astype(int)
analysis_df["SPEED_LIMIT_KMH"] = pd.to_numeric(
    analysis_df["Speed limit"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

CATEGORICAL_FEATURES = [
    "Weather", "Natural lighting", "Surface condition", "Road surface", "Alignment",
    "Type of location", "Road classification (admin)", "Signals operation", "Other traffic control",
    "School zone location", "School zone active", "Street type", "Urbanisation", "Conurbation 1", "LGA",
    "Day of week of crash", "Two-hour intervals", "Month of crash",
    "Primary permanent feature", "Primary hazardous feature", "Primary temporary feature",
]
NUMERIC_FEATURES = ["SPEED_LIMIT_KMH", "Latitude", "Longitude"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
for c in CATEGORICAL_FEATURES:
    analysis_df[c] = analysis_df[c].fillna("None / not recorded").astype(str)

TARGET = "KSI"
log("rows", len(analysis_df), "KSI rate", round(analysis_df.KSI.mean(), 4))

def make_pre(cats, nums):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), cats),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
    ])

train_df = analysis_df[analysis_df["Year of crash"].between(2020, 2022)].copy()
valid_df = analysis_df[analysis_df["Year of crash"] == 2023].copy()
test_df = analysis_df[analysis_df["Year of crash"] == 2024].copy()
devel_df = analysis_df[analysis_df["Year of crash"] <= 2023].copy()
split_tbl = pd.DataFrame([
    ("Train", "2020-2022", len(train_df), train_df[TARGET].mean()),
    ("Validation", "2023", len(valid_df), valid_df[TARGET].mean()),
    ("Test", "2024", len(test_df), test_df[TARGET].mean()),
], columns=["split", "years", "n", "ksi_rate"]).round(4)
split_tbl.to_csv(f"{OUT}/split_summary.csv", index=False)
log("\n" + split_tbl.to_string(index=False))

X_tr, y_tr = train_df[MODEL_FEATURES], train_df[TARGET]
X_va, y_va = valid_df[MODEL_FEATURES], valid_df[TARGET]
X_dev, y_dev = devel_df[MODEL_FEATURES], devel_df[TARGET]
X_te, y_te = test_df[MODEL_FEATURES], test_df[TARGET]

# ---- baselines (development data only) ----
def smoothed_rate(frame, keys, k=50):
    prior = frame[TARGET].mean()
    g = frame.groupby(keys)[TARGET].agg(["mean", "size"])
    g["smoothed"] = (g["mean"] * g["size"] + prior * k) / (g["size"] + k)
    return g["smoothed"], prior

def apply_rate(frame, table, prior, keys):
    idx = frame.set_index(keys).index
    return pd.Series(idx.map(table), index=frame.index).astype(float).fillna(prior).values

b0 = np.full(len(y_te), y_dev.mean())
t1, p1 = smoothed_rate(devel_df, ["LGA"]); b1 = apply_rate(test_df, t1, p1, ["LGA"])
k2 = ["LGA", "Speed limit", "Natural lighting"]
t2, p2 = smoothed_rate(devel_df, k2); b2 = apply_rate(test_df, t2, p2, k2)
hi = devel_df.loc[devel_df.SPEED_LIMIT_KMH >= 80, TARGET].mean()
lo = devel_df.loc[~(devel_df.SPEED_LIMIT_KMH >= 80), TARGET].mean()
b3 = np.where(test_df["SPEED_LIMIT_KMH"] >= 80, hi, lo)

def metrics(y, p):
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p)}

def candidates():
    return {
        "C1 Penalised logistic regression": Pipeline([("pre", make_pre(CATEGORICAL_FEATURES, NUMERIC_FEATURES)),
            ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=3000, solver="lbfgs"))]),
        "C2 Random forest": Pipeline([("pre", make_pre(CATEGORICAL_FEATURES, NUMERIC_FEATURES)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE))]),
        "C3 HistGradientBoosting": Pipeline([("pre", make_pre(CATEGORICAL_FEATURES, NUMERIC_FEATURES)),
            ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                                   l2_regularization=1.0, early_stopping=True,
                                                   validation_fraction=0.15, random_state=RANDOM_STATE))]),
    }

val_rows = []
for name, model in candidates().items():
    t = time.time(); model.fit(X_tr, y_tr)
    val_rows.append({"model": name, **metrics(y_va, model.predict_proba(X_va)[:, 1]),
                     "fit_seconds": round(time.time() - t, 1)})
    log("validated", name)
val_tbl = pd.DataFrame(val_rows).round(4)
val_tbl.to_csv(f"{OUT}/validation_model_selection.csv", index=False)
log("\n" + val_tbl.to_string(index=False))

# Selection rule: highest validation ROC-AUC, but candidates within 0.005 of the best are
# treated as tied (smaller than the bootstrap CI half-width) and the tie goes to the model
# that can be exported to and evaluated in the static browser dashboard. The random forest
# (300 deep trees) cannot be shipped to a GitHub Pages site; gradient boosting can.
DEPLOYABLE = ["C3 HistGradientBoosting", "C1 Penalised logistic regression"]
top_auc = val_tbl["ROC_AUC"].max()
tied = val_tbl[val_tbl["ROC_AUC"] >= top_auc - 0.005]["model"].tolist()
best_name = next((m for m in DEPLOYABLE if m in tied),
                 val_tbl.sort_values("ROC_AUC", ascending=False).iloc[0]["model"])
log("tied within 0.005:", tied, "-> selected:", best_name)

# held-out 2024 results for every candidate (refit on 2020-2023), for transparency
all_rows = []
for name, model in candidates().items():
    model.fit(X_dev, y_dev)
    all_rows.append({"model": name, **metrics(y_te, model.predict_proba(X_te)[:, 1])})
all_tbl = pd.DataFrame(all_rows).round(4)
all_tbl.to_csv(f"{OUT}/test_results_all_candidates.csv", index=False)
log("\n" + all_tbl.to_string(index=False))
final_model = candidates()[best_name].fit(X_dev, y_dev)
p_test_raw = final_model.predict_proba(X_te)[:, 1]
# Sigmoid (Platt) calibration instead of the notebook's isotonic: equal ROC-AUC and ECE on 2024, but
# isotonic is a step function, so many different scenarios received an identical probability and the
# scenario explorer could not show the effect of changing one condition.
CALIB = "sigmoid"
calibrated = CalibratedClassifierCV(candidates()[best_name], method=CALIB, cv=3)
calibrated.fit(X_dev, y_dev)
p_test_cal = calibrated.predict_proba(X_te)[:, 1]

rows = [
    {"method": "B0 Prior rate", **metrics(y_te, b0)},
    {"method": "B1 Historical LGA KSI rate", **metrics(y_te, b1)},
    {"method": "B2 LGA x speed x lighting cell rate", **metrics(y_te, b2)},
    {"method": "B3 Speed-limit rule (>=80 km/h)", **metrics(y_te, b3)},
    {"method": f"{best_name} (uncalibrated)", **metrics(y_te, p_test_raw)},
    {"method": f"{best_name} + {CALIB} calibration", **metrics(y_te, p_test_cal)},
]
test_tbl = pd.DataFrame(rows).round(4)
test_tbl["ROC_AUC_gain_vs_B2"] = (test_tbl["ROC_AUC"] - test_tbl.loc[2, "ROC_AUC"]).round(4)
test_tbl.to_csv(f"{OUT}/test_results_vs_baselines.csv", index=False)
log("\n" + test_tbl.to_string(index=False))

rng = np.random.default_rng(RANDOM_STATE)
boot = [roc_auc_score(y_te.values[i], p_test_cal[i]) for i in
        (rng.integers(0, len(y_te), len(y_te)) for _ in range(300))]
ci = np.percentile(boot, [2.5, 97.5]).round(4)

frac_pos, mean_pred = calibration_curve(y_te, p_test_cal, n_bins=10, strategy="quantile")
ece = float(np.mean(np.abs(frac_pos - mean_pred)))

order = np.argsort(-p_test_cal); yv = y_te.values[order]
rank_tbl = pd.DataFrame([{"top_k_percent": int(f * 100), "n_sites": int(len(yv) * f),
                          "precision_in_top_k": yv[:int(len(yv) * f)].mean(),
                          "recall_of_all_ksi": yv[:int(len(yv) * f)].sum() / yv.sum(),
                          "lift_vs_random": yv[:int(len(yv) * f)].mean() / yv.mean()}
                         for f in [0.05, 0.10, 0.20, 0.30, 0.50]]).round(4)
rank_tbl.to_csv(f"{OUT}/topk_ranking_performance.csv", index=False)
log("\n" + rank_tbl.to_string(index=False))

# validation-design robustness (as notebook B8)
cv_rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
s = cross_val_score(candidates()[best_name], X_dev, y_dev, cv=skf, scoring="roc_auc", n_jobs=1)
cv_rows.append({"design": "5-fold stratified CV (random)", "mean_ROC_AUC": s.mean(), "std": s.std()})
s2 = cross_val_score(candidates()[best_name], X_dev, y_dev, cv=GroupKFold(n_splits=5),
                     groups=devel_df["LGA"], scoring="roc_auc", n_jobs=1)
cv_rows.append({"design": "5-fold grouped CV (held-out LGAs)", "mean_ROC_AUC": s2.mean(), "std": s2.std()})
roll = []
for tr_end, va_yr in [(2020, 2021), (2021, 2022), (2022, 2023)]:
    a = analysis_df[analysis_df["Year of crash"] <= tr_end]; b = analysis_df[analysis_df["Year of crash"] == va_yr]
    m = candidates()[best_name].fit(a[MODEL_FEATURES], a[TARGET])
    roll.append(roc_auc_score(b[TARGET], m.predict_proba(b[MODEL_FEATURES])[:, 1]))
cv_rows.append({"design": "Rolling-origin temporal CV", "mean_ROC_AUC": np.mean(roll), "std": np.std(roll)})
cv_tbl = pd.DataFrame(cv_rows).round(4)
cv_tbl.to_csv(f"{OUT}/cv_robustness.csv", index=False)
log("\n" + cv_tbl.to_string(index=False))

# permutation importance (as notebook B13)
samp = test_df.sample(n=min(8000, len(test_df)), random_state=RANDOM_STATE)
perm = permutation_importance(final_model, samp[MODEL_FEATURES], samp[TARGET],
                              n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=1)
imp = pd.DataFrame({"feature": MODEL_FEATURES, "auc_drop": perm.importances_mean,
                    "std": perm.importances_std}).sort_values("auc_drop", ascending=False).round(5)
imp.to_csv(f"{OUT}/permutation_importance.csv", index=False)
log("\n" + imp.head(12).to_string(index=False))

# error analysis (as notebook B12)
err = test_df.copy(); err["p_pred"] = p_test_cal; err["y_true"] = y_te.values
def subgroup_report(frame, col, min_n=200):
    out = []
    for level, g in frame.groupby(col):
        if len(g) < min_n or g["y_true"].nunique() < 2:
            continue
        out.append({"segment": col, "level": str(level), "n": len(g),
                    "observed_rate": g["y_true"].mean(), "mean_predicted": g["p_pred"].mean(),
                    "calibration_gap": g["p_pred"].mean() - g["y_true"].mean(),
                    "ROC_AUC": roc_auc_score(g["y_true"], g["p_pred"])})
    return pd.DataFrame(out)
subgroups = pd.concat([subgroup_report(err, c) for c in
                       ["Urbanisation", "Speed limit", "Natural lighting", "Weather",
                        "Road classification (admin)", "Type of location"]], ignore_index=True).round(4)
subgroups.to_csv(f"{OUT}/subgroup_performance.csv", index=False)
lga_perf = subgroup_report(err, "LGA", min_n=150).round(4).sort_values("ROC_AUC")
lga_perf.to_csv(f"{OUT}/lga_performance.csv", index=False)

# darkness check: does KSI keep the expected direction?
direction = err.groupby("Natural lighting").agg(n=("y_true", "size"), observed=("y_true", "mean"),
                                                predicted=("p_pred", "mean")).round(4)
direction.to_csv(f"{OUT}/lighting_direction_check.csv")
log("\n" + direction.to_string())

summary = {"target": TARGET, "selected_model": best_name,
           "test_roc_auc": round(float(roc_auc_score(y_te, p_test_cal)), 4),
           "roc_auc_95ci": [float(ci[0]), float(ci[1])],
           "test_pr_auc": round(float(average_precision_score(y_te, p_test_cal)), 4),
           "test_brier": round(float(brier_score_loss(y_te, p_test_cal)), 4),
           "expected_calibration_error": round(ece, 4),
           "b2_roc_auc": float(test_tbl.loc[2, "ROC_AUC"]),
           "prevalence_test": round(float(y_te.mean()), 4),
           "prevalence_dev": round(float(y_dev.mean()), 4),
           "top10_precision": float(rank_tbl.loc[1, "precision_in_top_k"]),
           "top10_lift": float(rank_tbl.loc[1, "lift_vs_random"]),
           "grouped_cv_auc": float(cv_tbl.loc[1, "mean_ROC_AUC"]),
           "rolling_cv_auc": float(cv_tbl.loc[2, "mean_ROC_AUC"])}
json.dump(summary, open(f"{OUT}/results_summary.json", "w"), indent=2)
log(json.dumps(summary, indent=2))

with open(f"{OUT}/artefacts.pkl", "wb") as f:
    pickle.dump({"calibrated": calibrated, "final_model": final_model, "err": err,
                 "analysis_df": analysis_df, "b2": b2, "MODEL_FEATURES": MODEL_FEATURES,
                 "CATEGORICAL_FEATURES": CATEGORICAL_FEATURES, "NUMERIC_FEATURES": NUMERIC_FEATURES,
                 "imp": imp, "lga_perf": lga_perf, "summary": summary, "rank_tbl": rank_tbl}, f)
log("done")
