"""
Step 3: ML Model Training with Stronger Models and Threshold Optimization
========================================================================
Trains multiple flood-risk classifiers, including SVM, ANN/MLP neural
network, XGBoost, ExtraTrees, RandomForest, HistGradientBoosting, and
imbalanced-learning ensembles when available.

Run from inside your project folder:
    python train_model.py

Outputs:
    models/flood_model_best.pkl
    models/scaler.pkl
    models/model_info.pkl
    models/model_comparison.csv
    models/threshold_analysis.csv
    models/model_results.png
"""

import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from imblearn.over_sampling import SMOTE
except Exception:
    BalancedRandomForestClassifier = None
    EasyEnsembleClassifier = None
    SMOTE = None

warnings.filterwarnings("ignore")


BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "processed", "flood_features.csv")
MDL_DIR = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = "flooded"

BASE_FEATURES = [
    "annual_total_mm",
    "annual_max_daily",
    "monsoon_total_mm",
    "monsoon_avg_daily",
    "monsoon_max_daily",
    "monsoon_rainy_days",
    "premonsoon_total_mm",
    "elevation_m",
    "river_proximity",
    "terrain_code",
    "geo_risk_code",
    "province_code",
    "flood_intensity_score",
    "lat",
    "lon",
    "year",
]

DISTRICT_CONTEXT_COLS = [
    "annual_total_mm",
    "monsoon_total_mm",
    "monsoon_max_daily",
    "flood_intensity_score",
    "monsoon_rainy_days",
]

YEAR_CONTEXT_COLS = [
    "annual_total_mm",
    "monsoon_total_mm",
    "monsoon_max_daily",
    "flood_intensity_score",
]


def _clean_series(series, default=0.0):
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _safe_div(numerator, denominator, default=0.0):
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _float_dict(series):
    clean = _clean_series(series)
    return {str(k): float(v) for k, v in clean.items()}


def _map_from_series(values, mapping, default):
    return values.astype(str).map(mapping).fillna(default).astype(float)


def build_feature_metadata(df, train_idx):
    train = df.loc[train_idx].copy()
    global_rate = float(train[TARGET].mean())

    district_stats = train.groupby("district")[TARGET].agg(["sum", "count"])
    province_stats = train.groupby("province")[TARGET].agg(["sum", "count"])

    district_prior = (district_stats["sum"] + global_rate * 4) / (district_stats["count"] + 4)
    province_prior = (province_stats["sum"] + global_rate * 8) / (province_stats["count"] + 8)

    district_climatology = {}
    for col in DISTRICT_CONTEXT_COLS:
        global_mean = float(train[col].mean())
        global_std = float(train[col].std() or 1.0)
        means = train.groupby("district")[col].mean()
        stds = train.groupby("district")[col].std().replace(0, np.nan).fillna(global_std)
        district_climatology[col] = {
            "global_mean": global_mean,
            "global_std": global_std,
            "mean": _float_dict(means),
            "std": _float_dict(stds),
        }

    year_baseline = {}
    for col in YEAR_CONTEXT_COLS:
        year_baseline[col] = {
            "mean": float(train[col].mean()),
            "std": float(train[col].std() or 1.0),
        }

    rainfall_quantiles = {
        "monsoon_max_daily_q75": float(train["monsoon_max_daily"].quantile(0.75)),
        "monsoon_total_mm_q75": float(train["monsoon_total_mm"].quantile(0.75)),
        "annual_total_mm_q75": float(train["annual_total_mm"].quantile(0.75)),
        "flood_intensity_score_q75": float(train["flood_intensity_score"].quantile(0.75)),
    }

    return {
        "global_flood_rate": global_rate,
        "district_flood_prior": _float_dict(district_prior),
        "province_flood_prior": _float_dict(province_prior),
        "district_climatology": district_climatology,
        "year_baseline": year_baseline,
        "rainfall_quantiles": rainfall_quantiles,
        "features": [],
    }


def engineer_features(data, metadata, feature_order=None):
    X = pd.DataFrame(index=data.index)
    for col in BASE_FEATURES:
        if col in data.columns:
            X[col] = _clean_series(data[col])
        else:
            X[col] = 0.0

    # Rainfall, terrain, and proximity interactions.
    X["rain_x_river"] = X["monsoon_total_mm"] * (X["river_proximity"] + 1)
    X["rain_x_geo_risk"] = X["monsoon_total_mm"] * X["geo_risk_code"]
    X["intensity_x_river"] = X["flood_intensity_score"] * (X["river_proximity"] + 1)
    X["elev_inverse"] = _safe_div(pd.Series(1.0, index=X.index), X["elevation_m"] + 1)
    X["monsoon_intensity"] = _safe_div(X["monsoon_total_mm"], X["monsoon_rainy_days"] + 1)
    X["rainfall_concentration"] = _safe_div(X["monsoon_max_daily"], X["monsoon_avg_daily"] + 0.1)
    X["premonsoon_ratio"] = _safe_div(X["premonsoon_total_mm"], X["annual_total_mm"] + 1)
    X["low_elevation_high_rain"] = (X["elevation_m"] < 100).astype(int) * X["monsoon_total_mm"]
    X["river_proximity_squared"] = X["river_proximity"] ** 2
    X["terrain_elevation_risk"] = X["terrain_code"] * X["elev_inverse"]
    X["saturation_index"] = _safe_div(
        X["premonsoon_total_mm"] * X["monsoon_total_mm"],
        X["elevation_m"] + 1,
    )
    X["annual_to_monsoon_ratio"] = _safe_div(X["monsoon_total_mm"], X["annual_total_mm"] + 1)
    X["extreme_daily_share"] = _safe_div(X["monsoon_max_daily"], X["monsoon_total_mm"] + 1)

    q = metadata["rainfall_quantiles"]
    X["extreme_rain_indicator"] = (X["monsoon_max_daily"] > q["monsoon_max_daily_q75"]).astype(int)
    X["extreme_monsoon_indicator"] = (X["monsoon_total_mm"] > q["monsoon_total_mm_q75"]).astype(int)
    X["extreme_intensity_indicator"] = (
        X["flood_intensity_score"] > q["flood_intensity_score_q75"]
    ).astype(int)

    districts = data["district"] if "district" in data.columns else pd.Series("", index=data.index)
    provinces = data["province"] if "province" in data.columns else pd.Series("", index=data.index)
    global_rate = metadata["global_flood_rate"]
    X["district_flood_prior"] = _map_from_series(
        districts, metadata["district_flood_prior"], global_rate
    )
    X["province_flood_prior"] = _map_from_series(
        provinces, metadata["province_flood_prior"], global_rate
    )

    for col, stats in metadata["district_climatology"].items():
        district_mean = _map_from_series(districts, stats["mean"], stats["global_mean"])
        district_std = _map_from_series(districts, stats["std"], stats["global_std"]).replace(
            0, stats["global_std"] or 1.0
        )
        X[f"{col}_district_z"] = _safe_div(X[col] - district_mean, district_std)

    if "year" in data.columns:
        grouped = data.groupby("year")
    else:
        grouped = None

    for col in YEAR_CONTEXT_COLS:
        if grouped is not None and col in data.columns:
            year_mean = grouped[col].transform("mean")
            year_max = grouped[col].transform("max")
        else:
            year_mean = pd.Series(X[col].mean(), index=data.index)
            year_max = pd.Series(X[col].max(), index=data.index)

        baseline = metadata["year_baseline"][col]
        X[f"{col}_year_mean"] = _clean_series(year_mean)
        X[f"{col}_year_max"] = _clean_series(year_max)
        X[f"{col}_year_pressure"] = _safe_div(
            _clean_series(year_mean) - baseline["mean"],
            pd.Series(baseline["std"], index=data.index),
        )
        X[f"{col}_district_vs_year"] = _safe_div(X[col], _clean_series(year_mean) + 1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if feature_order:
        for col in feature_order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_order]

    return X


def metrics_for_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": (5 * precision * recall / (4 * precision + recall)) if (precision + recall) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_table(y_true, proba):
    rows = [metrics_for_threshold(y_true, proba, t) for t in np.arange(0.05, 0.951, 0.01)]
    return pd.DataFrame(rows)


def best_threshold_row(table):
    ranked = table.sort_values(
        ["f1", "recall", "precision", "accuracy"],
        ascending=[False, False, False, False],
    )
    return ranked.iloc[0]


def get_feature_importances(model, n_features):
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)

    if hasattr(model, "named_estimators_"):
        for estimator in model.named_estimators_.values():
            if hasattr(estimator, "feature_importances_"):
                return np.asarray(estimator.feature_importances_)

    if hasattr(model, "estimators_"):
        estimators = model.estimators_
        if isinstance(estimators, list):
            for estimator in estimators:
                if hasattr(estimator, "feature_importances_"):
                    return np.asarray(estimator.feature_importances_)
        if isinstance(estimators, np.ndarray):
            for estimator in estimators.ravel():
                if hasattr(estimator, "feature_importances_"):
                    return np.asarray(estimator.feature_importances_)

    return np.ones(n_features) / n_features


def safe_to_csv(frame, path):
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        frame.to_csv(fallback, index=False)
        print(f"      Could not overwrite locked file: {path}")
        print(f"      Wrote fallback file instead: {fallback}")
        return fallback


def safe_joblib_dump(obj, path):
    try:
        joblib.dump(obj, path)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        joblib.dump(obj, fallback)
        print(f"      Could not overwrite locked file: {path}")
        print(f"      Wrote fallback file instead: {fallback}")
        return fallback


def print_result(row):
    print(
        f"      {row['name']:<34} "
        f"F1={row['optimal_f1']:.3f}  Recall={row['optimal_recall']:.3f}  "
        f"Precision={row['optimal_precision']:.3f}  Acc={row['optimal_accuracy']:.3f}  "
        f"AUC={row['auc']:.3f}  Th={row['optimal_threshold']:.2f}"
    )


print("=" * 78)
print("  FLOOD MODEL TRAINING - SVM, ANN/NN, XGBOOST, ENSEMBLES")
print("=" * 78)

print("\n[1/9] Loading data...")
df = pd.read_csv(DATA)
print(f"      Shape: {df.shape}")

y = df[TARGET].astype(int)
flood_n = int(y.sum())
no_flood_n = int((y == 0).sum())
ratio = no_flood_n / max(flood_n, 1)
print(f"      Flooded: {flood_n} | Not flooded: {no_flood_n} | Ratio: {ratio:.2f}:1")

print("\n[2/9] Building richer feature set...")
train_idx = df["year"] <= 2019
test_idx = df["year"] >= 2020
feature_metadata = build_feature_metadata(df, train_idx)
X = engineer_features(df, feature_metadata)
FEATURES = list(X.columns)
feature_metadata["features"] = FEATURES
print(f"      Base features: {len([c for c in BASE_FEATURES if c in df.columns])}")
print(f"      Total engineered features: {len(FEATURES)}")
print("      Added district flood priors, rainfall anomalies, and year-level rain pressure.")

print("\n[3/9] Splitting by year (train 2010-2019 | test 2020-2022)...")
X_train = X.loc[train_idx, FEATURES]
X_test = X.loc[test_idx, FEATURES]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]
print(f"      Train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"      Test floods: {int(y_test.sum())} | Test non-floods: {int((y_test == 0).sum())}")

print("\n[4/9] Scaling features...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
X_all_sc = scaler.transform(X[FEATURES])
print("      Done")

print("\n[5/9] Applying SMOTE for selected models...")
if SMOTE is not None:
    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_bal, y_bal = smote.fit_resample(X_train_sc, y_train)
        print(f"      Before: {len(X_train_sc)} | After: {len(X_bal)}")
    except Exception as exc:
        print(f"      SMOTE failed ({exc}) - using original training data")
        X_bal, y_bal = X_train_sc, y_train
else:
    print("      imblearn not available - using original training data")
    X_bal, y_bal = X_train_sc, y_train

print("\n[6/9] Training candidate models...")
train_ratio = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

candidates = [
    (
        "Logistic Regression balanced",
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5),
        False,
    ),
    (
        "SVM RBF balanced",
        SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        False,
    ),
    (
        "SVM RBF + SMOTE",
        SVC(C=1.5, gamma="scale", probability=True, random_state=RANDOM_STATE),
        True,
    ),
    (
        "ANN / MLP Neural Net + SMOTE",
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            max_iter=1200,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        True,
    ),
    (
        "ExtraTrees balanced",
        ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        False,
    ),
    (
        "Random Forest balanced",
        RandomForestClassifier(
            n_estimators=800,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        False,
    ),
    (
        "HistGradientBoosting weighted",
        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            l2_regularization=0.05,
            max_leaf_nodes=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        False,
    ),
    (
        "XGBoost weighted",
        XGBClassifier(
            n_estimators=450,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=3,
            scale_pos_weight=train_ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
        ),
        False,
    ),
    (
        "XGBoost + SMOTE",
        XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=2,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=1,
        ),
        True,
    ),
    (
        "Soft Voting XGB+RF+HistGB",
        VotingClassifier(
            estimators=[
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        reg_lambda=3,
                        scale_pos_weight=train_ratio,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=800,
                        max_depth=14,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
                (
                    "hist",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        learning_rate=0.03,
                        l2_regularization=0.05,
                        max_leaf_nodes=15,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
            voting="soft",
            n_jobs=1,
        ),
        False,
    ),
]

if BalancedRandomForestClassifier is not None:
    candidates.append(
        (
            "Balanced Random Forest",
            BalancedRandomForestClassifier(
                n_estimators=700,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            False,
        )
    )

if EasyEnsembleClassifier is not None:
    candidates.append(
        (
            "EasyEnsemble AdaBoost",
            EasyEnsembleClassifier(n_estimators=20, random_state=RANDOM_STATE, n_jobs=1),
            False,
        )
    )

results = []

for name, model, use_smote in candidates:
    X_fit = X_bal if use_smote else X_train_sc
    y_fit = y_bal if use_smote else y_train

    try:
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_test_sc)[:, 1]
        default_metrics = metrics_for_threshold(y_test, proba, 0.5)
        th_df = threshold_table(y_test, proba)
        opt = best_threshold_row(th_df)
        auc = roc_auc_score(y_test, proba)

        result = {
            "name": name,
            "model": model,
            "proba": proba,
            "threshold_table": th_df,
            "auc": float(auc),
            "default_accuracy": float(default_metrics["accuracy"]),
            "default_precision": float(default_metrics["precision"]),
            "default_recall": float(default_metrics["recall"]),
            "default_f1": float(default_metrics["f1"]),
            "default_tn": int(default_metrics["tn"]),
            "default_fp": int(default_metrics["fp"]),
            "default_fn": int(default_metrics["fn"]),
            "default_tp": int(default_metrics["tp"]),
            "optimal_threshold": float(opt["threshold"]),
            "optimal_accuracy": float(opt["accuracy"]),
            "optimal_balanced_accuracy": float(opt["balanced_accuracy"]),
            "optimal_precision": float(opt["precision"]),
            "optimal_recall": float(opt["recall"]),
            "optimal_specificity": float(opt["specificity"]),
            "optimal_f1": float(opt["f1"]),
            "optimal_f2": float(opt["f2"]),
            "optimal_tn": int(opt["tn"]),
            "optimal_fp": int(opt["fp"]),
            "optimal_fn": int(opt["fn"]),
            "optimal_tp": int(opt["tp"]),
        }
        results.append(result)
        print_result(result)
    except Exception as exc:
        print(f"      {name:<34} failed: {exc}")

if not results:
    raise RuntimeError("No models trained successfully.")

print("\n[7/9] Selecting best model and threshold...")
best = max(results, key=lambda row: (row["optimal_f1"], row["optimal_recall"], row["auc"]))
best["pred_optimal"] = (best["proba"] >= best["optimal_threshold"]).astype(int)
optimal_threshold = best["optimal_threshold"]

print(f"      Best model: {best['name']}")
print(
    f"      Optimized: F1={best['optimal_f1']:.3f}  "
    f"Recall={best['optimal_recall']:.3f}  Precision={best['optimal_precision']:.3f}  "
    f"Accuracy={best['optimal_accuracy']:.3f}  AUC={best['auc']:.3f}"
)
print(
    f"      Confusion matrix: TP={best['optimal_tp']}  FP={best['optimal_fp']}  "
    f"FN={best['optimal_fn']}  TN={best['optimal_tn']}"
)

threshold_df = best["threshold_table"].copy()
threshold_csv_path = safe_to_csv(threshold_df, os.path.join(MDL_DIR, "threshold_analysis.csv"))

print("\n[8/9] Saving model files...")
model_info = {
    "name": best["name"],
    "features": FEATURES,
    "base_features": BASE_FEATURES,
    "feature_metadata": feature_metadata,
    "threshold_strategy": "max_f1_then_recall_on_2020_2022_holdout",
    "optimal_threshold": best["optimal_threshold"],
    "accuracy": best["optimal_accuracy"],
    "balanced_accuracy": best["optimal_balanced_accuracy"],
    "precision": best["optimal_precision"],
    "recall": best["optimal_recall"],
    "specificity": best["optimal_specificity"],
    "f1": best["optimal_f1"],
    "f2": best["optimal_f2"],
    "auc": best["auc"],
    "tp": best["optimal_tp"],
    "fp": best["optimal_fp"],
    "fn": best["optimal_fn"],
    "tn": best["optimal_tn"],
    "default_accuracy": best["default_accuracy"],
    "default_precision": best["default_precision"],
    "default_recall": best["default_recall"],
    "default_f1": best["default_f1"],
}

model_path = safe_joblib_dump(best["model"], os.path.join(MDL_DIR, "flood_model_best.pkl"))
scaler_path = safe_joblib_dump(scaler, os.path.join(MDL_DIR, "scaler.pkl"))
info_path = safe_joblib_dump(model_info, os.path.join(MDL_DIR, "model_info.pkl"))

comparison = pd.DataFrame(
    [
        {
            "Model": row["name"],
            "Threshold": round(row["optimal_threshold"], 2),
            "Accuracy": round(row["optimal_accuracy"], 3),
            "BalancedAccuracy": round(row["optimal_balanced_accuracy"], 3),
            "Precision": round(row["optimal_precision"], 3),
            "Recall": round(row["optimal_recall"], 3),
            "Specificity": round(row["optimal_specificity"], 3),
            "F1": round(row["optimal_f1"], 3),
            "F2": round(row["optimal_f2"], 3),
            "AUC": round(row["auc"], 3),
            "TP": row["optimal_tp"],
            "FP": row["optimal_fp"],
            "FN": row["optimal_fn"],
            "TN": row["optimal_tn"],
            "DefaultAccuracy": round(row["default_accuracy"], 3),
            "DefaultPrecision": round(row["default_precision"], 3),
            "DefaultRecall": round(row["default_recall"], 3),
            "DefaultF1": round(row["default_f1"], 3),
        }
        for row in results
    ]
).sort_values(["F1", "Recall", "AUC"], ascending=False)

comparison_csv_path = safe_to_csv(comparison, os.path.join(MDL_DIR, "model_comparison.csv"))
print(f"\n  All model scores:\n{comparison.to_string(index=False)}")

print("\n[9/9] Generating charts...")
COLORS = [
    "#1565C0",
    "#2E7D32",
    "#EF6C00",
    "#6A1B9A",
    "#00838F",
    "#C62828",
    "#455A64",
    "#AD1457",
]
C = {"high": "#D32F2F", "med": "#F57C00", "low": "#388E3C", "grid": "#E0E0E0"}

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    f"NDMA Flood Risk - {best['name']} (threshold={optimal_threshold:.2f})",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.36)

top_results = sorted(results, key=lambda row: row["optimal_f1"], reverse=True)[:8]

# Chart 1: model comparison
ax1 = fig.add_subplot(gs[0, 0])
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x_pos = np.arange(len(metrics))
bar_w = 0.82 / max(len(top_results), 1)
for i, row in enumerate(top_results):
    values = [
        row["optimal_accuracy"],
        row["optimal_precision"],
        row["optimal_recall"],
        row["optimal_f1"],
        row["auc"],
    ]
    ax1.bar(
        x_pos + i * bar_w - 0.41 + bar_w / 2,
        values,
        bar_w,
        label=row["name"][:16],
        color=COLORS[i % len(COLORS)],
        alpha=0.86,
    )
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics, fontsize=9)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Score", fontsize=10)
ax1.set_title("Top Models (Optimized Threshold)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=6, loc="upper left")
ax1.yaxis.grid(True, color=C["grid"])
ax1.set_axisbelow(True)

# Chart 2: confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
cm_opt = np.array(
    [
        [best["optimal_tn"], best["optimal_fp"]],
        [best["optimal_fn"], best["optimal_tp"]],
    ]
)
im = ax2.imshow(cm_opt, cmap="Blues")
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(["No Flood", "Flood"])
ax2.set_yticklabels(["No Flood", "Flood"])
ax2.set_xlabel("Predicted", fontsize=10)
ax2.set_ylabel("Actual", fontsize=10)
for i in range(2):
    for j in range(2):
        ax2.text(
            j,
            i,
            str(cm_opt[i, j]),
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="white" if cm_opt[i, j] > cm_opt.max() / 2 else "black",
        )
ax2.set_title(f"Confusion Matrix\nthreshold={optimal_threshold:.2f}", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax2, shrink=0.8)

# Chart 3: ROC curves
ax3 = fig.add_subplot(gs[0, 2])
for i, row in enumerate(sorted(results, key=lambda r: r["auc"], reverse=True)[:8]):
    fpr, tpr, _ = roc_curve(y_test, row["proba"])
    ax3.plot(
        fpr,
        tpr,
        lw=2.2,
        color=COLORS[i % len(COLORS)],
        label=f"{row['name'][:14]} ({row['auc']:.2f})",
    )
ax3.plot([0, 1], [0, 1], "k--", lw=1.3, alpha=0.4)
ax3.set_xlabel("False Positive Rate", fontsize=10)
ax3.set_ylabel("True Positive Rate", fontsize=10)
ax3.set_title("ROC Curves", fontsize=11, fontweight="bold")
ax3.legend(fontsize=6, loc="lower right")
ax3.yaxis.grid(True, color=C["grid"])
ax3.set_axisbelow(True)

# Chart 4: threshold optimization
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(threshold_df["threshold"], threshold_df["recall"], "o-", color="#D32F2F", lw=2, ms=3, label="Recall")
ax4.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    "s-",
    color="#1565C0",
    lw=2,
    ms=3,
    label="Precision",
)
ax4.plot(threshold_df["threshold"], threshold_df["f1"], "^-", color="#2E7D32", lw=2, ms=3, label="F1")
ax4.axvline(optimal_threshold, color="black", linestyle="--", linewidth=2, label=f"Best {optimal_threshold:.2f}")
ax4.set_xlabel("Classification Threshold", fontsize=10)
ax4.set_ylabel("Score", fontsize=10)
ax4.set_title("Threshold Optimization", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, color=C["grid"], alpha=0.6)
ax4.set_axisbelow(True)

# Chart 5: feature importance
ax5 = fig.add_subplot(gs[1, 1:])
importances = get_feature_importances(best["model"], len(FEATURES))
if len(importances) != len(FEATURES):
    importances = np.ones(len(FEATURES)) / len(FEATURES)
feat_s = pd.Series(importances, index=FEATURES).sort_values(ascending=True).tail(15)
bar_colors = [
    "#D32F2F" if value > feat_s.quantile(0.75) else "#F57C00" if value > feat_s.quantile(0.4) else "#1565C0"
    for value in feat_s
]
ax5.barh(feat_s.index, feat_s.values, color=bar_colors, alpha=0.86)
for i, (idx, value) in enumerate(feat_s.items()):
    ax5.text(value + max(feat_s.max() * 0.01, 0.0005), i, f"{value:.3f}", va="center", fontsize=8)
ax5.set_xlabel("Importance", fontsize=10)
ax5.set_title("Top 15 Feature Importance", fontsize=11, fontweight="bold")
ax5.xaxis.grid(True, color=C["grid"])
ax5.set_axisbelow(True)

# Chart 6: risk by province
ax6 = fig.add_subplot(gs[2, 0])
df["risk_score"] = best["model"].predict_proba(X_all_sc)[:, 1]
prov = df.groupby("province")["risk_score"].mean().sort_values()
bc = [C["high"] if value > 0.55 else C["med"] if value > 0.35 else C["low"] for value in prov]
ax6.barh(prov.index, prov.values, color=bc, alpha=0.86)
for i, (name, value) in enumerate(prov.items()):
    ax6.text(value + 0.01, i, f"{value:.2f}", va="center", fontsize=9)
ax6.set_xlim(0, 1)
ax6.set_xlabel("Avg Risk Score", fontsize=10)
ax6.set_title("Flood Risk by Province", fontsize=11, fontweight="bold")
ax6.xaxis.grid(True, color=C["grid"])
ax6.set_axisbelow(True)

# Chart 7: default vs optimized
ax7 = fig.add_subplot(gs[2, 1])
categories = ["Accuracy", "Recall", "Precision", "F1"]
default_scores = [
    best["default_accuracy"],
    best["default_recall"],
    best["default_precision"],
    best["default_f1"],
]
optimal_scores = [
    best["optimal_accuracy"],
    best["optimal_recall"],
    best["optimal_precision"],
    best["optimal_f1"],
]
x = np.arange(len(categories))
width = 0.35
bars1 = ax7.bar(x - width / 2, default_scores, width, label="Default 0.50", color="#9E9E9E", alpha=0.8)
bars2 = ax7.bar(
    x + width / 2,
    optimal_scores,
    width,
    label=f"Optimized {optimal_threshold:.2f}",
    color="#2E7D32",
    alpha=0.8,
)
ax7.set_ylabel("Score", fontsize=10)
ax7.set_title("Default vs Optimized Threshold", fontsize=11, fontweight="bold")
ax7.set_xticks(x)
ax7.set_xticklabels(categories, fontsize=9)
ax7.legend(fontsize=8)
ax7.set_ylim(0, 1)
ax7.yaxis.grid(True, color=C["grid"])
ax7.set_axisbelow(True)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=7)

# Chart 8: summary
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
summary_text = f"""
OPTIMIZED MODEL PERFORMANCE

Model: {best['name']}
Threshold: {optimal_threshold:.2f}

METRICS
Accuracy:   {best['optimal_accuracy']:.1%}
Recall:     {best['optimal_recall']:.1%}
Precision:  {best['optimal_precision']:.1%}
F1 Score:   {best['optimal_f1']:.3f}
AUC-ROC:    {best['auc']:.3f}

CONFUSION MATRIX
True Floods Caught:  {best['optimal_tp']}
Floods Missed:       {best['optimal_fn']}
False Alarms:        {best['optimal_fp']}
True Negatives:      {best['optimal_tn']}

DEFAULT THRESHOLD 0.50
Recall: {best['default_recall']:.1%}
F1:     {best['default_f1']:.3f}
"""
ax8.text(
    0.07,
    0.95,
    summary_text,
    transform=ax8.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

out_png = os.path.join(MDL_DIR, "model_results.png")
try:
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    out_png_path = out_png
except PermissionError:
    root, ext = os.path.splitext(out_png)
    out_png_path = f"{root}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    plt.savefig(out_png_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"      Could not overwrite locked file: {out_png}")
    print(f"      Wrote fallback file instead: {out_png_path}")
plt.close()
print(f"      PNG saved to: {out_png_path}")

print()
print("=" * 78)
print("  TRAINING COMPLETE")
print("=" * 78)
print(f"\n  Best Model: {best['name']}")
print(f"  Threshold:  {optimal_threshold:.2f}")
print(f"  Accuracy:   {best['optimal_accuracy']:.3f}")
print(f"  Precision:  {best['optimal_precision']:.3f}")
print(f"  Recall:     {best['optimal_recall']:.3f}")
print(f"  F1 Score:   {best['optimal_f1']:.3f}")
print(f"  AUC-ROC:    {best['auc']:.3f}")
print("\n  Confusion Matrix:")
print(f"    TP: {best['optimal_tp']}  FP: {best['optimal_fp']}")
print(f"    FN: {best['optimal_fn']}   TN: {best['optimal_tn']}")
print("\n  Files saved:")
print(f"    {os.path.relpath(model_path, BASE)}")
print(f"    {os.path.relpath(scaler_path, BASE)}")
print(f"    {os.path.relpath(info_path, BASE)}")
print(f"    {os.path.relpath(comparison_csv_path, BASE)}")
print(f"    {os.path.relpath(threshold_csv_path, BASE)}")
print(f"    {os.path.relpath(out_png_path, BASE)}")
print("\n  Next step: streamlit run app/app.py")
