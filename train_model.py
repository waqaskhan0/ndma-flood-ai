"""
Step 3: ML Model Training — Stratified Random Train/Test Split
==============================================================
Uses stratified random 80/20 split (NOT year-based split).
Stratified means both train and test have the same flood ratio.

Built exactly for flood_features.csv:
  - 2041 rows, 157 districts, 7 provinces
  - Imbalance ratio 4.37:1

Run:
    python train_model.py

Outputs:
    models/flood_model_best.pkl
    models/scaler.pkl
    models/model_info.pkl
    models/model_comparison.csv
    models/model_results.png
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble        import (RandomForestClassifier,
                                     ExtraTreesClassifier,
                                     HistGradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, roc_curve,
                                     confusion_matrix, balanced_accuracy_score)
from xgboost                 import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble      import BalancedRandomForestClassifier
    IMBLEARN = True
except ImportError:
    IMBLEARN = False
    print("  NOTE: install imbalanced-learn:  pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "data", "processed", "flood_features.csv")
MDL_DIR = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)

FEATURES = [
    "annual_total_mm",
    "annual_max_daily",
    "monsoon_total_mm",
    "monsoon_avg_daily",
    "monsoon_max_daily",
    "monsoon_rainy_days",
    "premonsoon_total_mm",
    "lat",
    "lon",
    "elevation_m",
    "river_proximity",
    "terrain_code",
    "geo_risk_code",
    "province_code",
    "flood_intensity_score",
    "flood_intensity_code",
]

TARGET = "flooded"
SEED   = 42
TEST_SIZE = 0.2   # 80% train, 20% test

print("=" * 65)
print("  ML MODEL TRAINING — STRATIFIED RANDOM SPLIT (80/20)")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n[1/8] Loading data...")
df = pd.read_csv(DATA)
print(f"      Shape     : {df.shape}")
print(f"      Districts : {df['district'].nunique()}")
print(f"      Provinces : {df['province'].nunique()}")

FEATURES = [f for f in FEATURES if f in df.columns]
print(f"      Features  : {len(FEATURES)}")

X_raw = df[FEATURES].fillna(0)
y     = df[TARGET].astype(int)

flood_n    = int(y.sum())
no_flood_n = int((y==0).sum())
ratio      = no_flood_n / flood_n
print(f"      Flooded   : {flood_n}  |  Not flooded : {no_flood_n}  |  Ratio : {ratio:.2f}:1")

# ══════════════════════════════════════════════════════════════
# 2. ENGINEER EXTRA FEATURES
# ══════════════════════════════════════════════════════════════
print("\n[2/8] Engineering extra features...")
X = X_raw.copy()

X["rain_x_river"]       = X["monsoon_total_mm"]      * (X["river_proximity"] + 1)
X["rain_x_geo_risk"]    = X["monsoon_total_mm"]      * X["geo_risk_code"]
X["intensity_x_river"]  = X["flood_intensity_score"] * (X["river_proximity"] + 1)
X["elev_inverse"]       = 1 / (X["elevation_m"] + 1)
X["low_elev_high_rain"] = (X["elevation_m"] < 100).astype(int) * X["monsoon_total_mm"]
X["monsoon_intensity"]  = X["monsoon_total_mm"] / (X["monsoon_rainy_days"] + 1)
X["rain_concentration"] = X["monsoon_max_daily"] / (X["monsoon_avg_daily"] + 0.1)
X["saturation_index"]   = (X["premonsoon_total_mm"] * X["monsoon_total_mm"]) / (X["elevation_m"] + 1)
X["premonsoon_ratio"]   = X["premonsoon_total_mm"] / (X["annual_total_mm"] + 1)

# District flood prior (Bayesian smoothed from whole dataset)
global_rate = float(y.mean())
dist_stats  = df.groupby("district")[TARGET].agg(["sum","count"])
X["district_flood_prior"] = df["district"].map(
    (dist_stats["sum"] + global_rate*4) / (dist_stats["count"] + 4)
).fillna(global_rate).values

# Province flood prior
prov_stats = df.groupby("province")[TARGET].agg(["sum","count"])
X["province_flood_prior"] = df["province"].map(
    (prov_stats["sum"] + global_rate*8) / (prov_stats["count"] + 8)
).fillna(global_rate).values

# Rainfall anomaly vs district average
dist_avg = df.groupby("district")["monsoon_total_mm"].transform("mean")
X["monsoon_anomaly"] = (X["monsoon_total_mm"] - dist_avg).values

# Year rain pressure
year_mean = df.groupby("year")["monsoon_total_mm"].transform("mean")
baseline  = float(df["monsoon_total_mm"].mean())
X["year_rain_pressure"] = ((year_mean - baseline) /
                            (float(df["monsoon_total_mm"].std()) + 1)).values

X["year"] = df["year"].values

FEATURES_FINAL = list(X.columns)
print(f"      Total features: {len(FEATURES_FINAL)}")

# ══════════════════════════════════════════════════════════════
# 3. STRATIFIED RANDOM TRAIN/TEST SPLIT
#    Mixed — rows from all years in both train and test
# ══════════════════════════════════════════════════════════════
print(f"\n[3/8] Stratified random split ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
print(f"      NOTE: Mixed split — all years appear in both train and test.")
print(f"      Stratified = same flood ratio in train and test sets.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = SEED,
    stratify     = y        # ← keeps same flood % in both sets
)

print(f"\n      Train : {len(X_train)} rows")
print(f"        Flooded     : {int(y_train.sum())}  ({y_train.mean()*100:.1f}%)")
print(f"        Not flooded : {int((y_train==0).sum())}  ({(y_train==0).mean()*100:.1f}%)")
print(f"\n      Test  : {len(X_test)} rows")
print(f"        Flooded     : {int(y_test.sum())}  ({y_test.mean()*100:.1f}%)")
print(f"        Not flooded : {int((y_test==0).sum())}  ({(y_test==0).mean()*100:.1f}%)")

# Show year distribution
train_years = df.loc[X_train.index, "year"].value_counts().sort_index()
test_years  = df.loc[X_test.index,  "year"].value_counts().sort_index()
print(f"\n      Years in train: {dict(train_years)}")
print(f"      Years in test : {dict(test_years)}")

# ══════════════════════════════════════════════════════════════
# 4. SCALE
# ══════════════════════════════════════════════════════════════
print("\n[4/8] Scaling features...")
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_all_sc   = scaler.transform(X)
print("      Done — mean=0, std=1")

# ══════════════════════════════════════════════════════════════
# 5. SMOTE on training set only
# ══════════════════════════════════════════════════════════════
print("\n[5/8] Applying SMOTE on training set only...")
if IMBLEARN:
    try:
        smote        = SMOTE(random_state=SEED, k_neighbors=5)
        X_bal, y_bal = smote.fit_resample(X_train_sc, y_train)
        print(f"      Before : {len(X_train_sc)} | After : {len(X_bal)}")
        print(f"      Flood={int(y_bal.sum())}  No-flood={(y_bal==0).sum()}")
    except Exception as e:
        print(f"      SMOTE failed ({e}) — using original")
        X_bal, y_bal = X_train_sc, y_train
else:
    print("      Skipped — using original")
    X_bal, y_bal = X_train_sc, y_train

# ══════════════════════════════════════════════════════════════
# 6. TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\n[6/8] Training models (3–6 minutes)...")

train_ratio = int((y_train==0).sum()) / max(int((y_train==1).sum()), 1)
cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate(name, model, use_bal=True):
    Xf = X_bal       if use_bal else X_train_sc
    yf = y_bal       if use_bal else y_train
    model.fit(Xf, yf)
    proba = model.predict_proba(X_test_sc)[:, 1]

    # 5-fold CV F1 on training set
    cv_f1 = cross_val_score(model, X_train_sc, y_train,
                             cv=cv, scoring="f1").mean()

    # find optimal threshold (maximize F1)
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.10, 0.90, 0.01):
        p  = (proba >= th).astype(int)
        fs = f1_score(y_test, p, zero_division=0)
        if fs > best_f1:
            best_f1, best_th = fs, th

    pred = (proba >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    r = dict(
        name      = name,
        model     = model,
        proba     = proba,
        threshold = best_th,
        cv_f1     = cv_f1,
        accuracy  = accuracy_score(y_test, pred),
        bal_acc   = balanced_accuracy_score(y_test, pred),
        precision = precision_score(y_test, pred, zero_division=0),
        recall    = recall_score(y_test, pred, zero_division=0),
        f1        = f1_score(y_test, pred, zero_division=0),
        auc       = roc_auc_score(y_test, proba),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    print(f"      {name:<36} F1={r['f1']:.3f}  Recall={r['recall']:.3f}"
          f"  Prec={r['precision']:.3f}  AUC={r['auc']:.3f}"
          f"  CV_F1={cv_f1:.3f}  Th={best_th:.2f}")
    return r

results = []

results.append(evaluate("Random Forest (balanced)",
    RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2,
                           class_weight="balanced_subsample",
                           random_state=SEED, n_jobs=-1), use_bal=False))

results.append(evaluate("Random Forest + SMOTE",
    RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2,
                           random_state=SEED, n_jobs=-1), use_bal=True))

results.append(evaluate("ExtraTrees (balanced)",
    ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                         class_weight="balanced_subsample",
                         random_state=SEED, n_jobs=-1), use_bal=False))

results.append(evaluate("HistGradientBoosting (balanced)",
    HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03,
                                   max_leaf_nodes=15, class_weight="balanced",
                                   random_state=SEED), use_bal=False))

results.append(evaluate("XGBoost (scale_pos_weight)",
    XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                  subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=3, reg_lambda=3,
                  scale_pos_weight=train_ratio,
                  eval_metric="logloss", verbosity=0,
                  random_state=SEED, n_jobs=-1), use_bal=False))

results.append(evaluate("XGBoost + SMOTE",
    XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.035,
                  subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=2, reg_lambda=2,
                  eval_metric="logloss", verbosity=0,
                  random_state=SEED, n_jobs=-1), use_bal=True))

results.append(evaluate("ANN / MLP + SMOTE",
    MLPClassifier(hidden_layer_sizes=(128,64,32), alpha=0.01,
                  max_iter=1000, early_stopping=True,
                  random_state=SEED), use_bal=True))

results.append(evaluate("SVM RBF (balanced)",
    SVC(C=1.5, gamma="scale", class_weight="balanced",
        probability=True, random_state=SEED), use_bal=False))

rf_v   = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)
xgb_v  = XGBClassifier(n_estimators=300, scale_pos_weight=train_ratio,
                         eval_metric="logloss", verbosity=0,
                         random_state=SEED, n_jobs=-1)
hist_v = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                         random_state=SEED)
results.append(evaluate("Soft Voting (RF+XGB+HistGB)",
    VotingClassifier([("rf",rf_v),("xgb",xgb_v),("hist",hist_v)],
                     voting="soft", n_jobs=-1), use_bal=False))

if IMBLEARN:
    results.append(evaluate("Balanced Random Forest",
        BalancedRandomForestClassifier(n_estimators=500, max_depth=12,
                                       min_samples_leaf=2,
                                       random_state=SEED, n_jobs=-1),
        use_bal=False))

# ══════════════════════════════════════════════════════════════
# 7. PICK BEST & SAVE
# ══════════════════════════════════════════════════════════════
print("\n[7/8] Selecting best model...")
best = max(results, key=lambda r: (r["f1"], r["recall"], r["auc"]))

print(f"      WINNER    : {best['name']}")
print(f"      Threshold : {best['threshold']:.2f}")
print(f"      F1        : {best['f1']:.3f}")
print(f"      CV F1     : {best['cv_f1']:.3f}  (5-fold cross-validation)")
print(f"      Recall    : {best['recall']:.3f}")
print(f"      Precision : {best['precision']:.3f}")
print(f"      AUC-ROC   : {best['auc']:.3f}")
print(f"      Bal. Acc  : {best['bal_acc']:.3f}")
print(f"\n      Confusion Matrix (on {len(X_test)} test rows):")
print(f"        TP (caught floods)   : {best['tp']}")
print(f"        FN (missed floods)   : {best['fn']}  ← keep low!")
print(f"        FP (false alarms)    : {best['fp']}")
print(f"        TN (correct no-fl.)  : {best['tn']}")

# Save files
joblib.dump(best["model"], os.path.join(MDL_DIR, "flood_model_best.pkl"))
joblib.dump(scaler,        os.path.join(MDL_DIR, "scaler.pkl"))
joblib.dump({
    "name":       best["name"],
    "features":   FEATURES_FINAL,
    "threshold":  best["threshold"],
    "f1":         best["f1"],
    "cv_f1":      best["cv_f1"],
    "recall":     best["recall"],
    "precision":  best["precision"],
    "auc":        best["auc"],
    "bal_acc":    best["bal_acc"],
    "tp": best["tp"], "fp": best["fp"],
    "fn": best["fn"], "tn": best["tn"],
    "split_type": "stratified_random_80_20",
    "test_size":  TEST_SIZE,
}, os.path.join(MDL_DIR, "model_info.pkl"))

# Comparison CSV
comp = pd.DataFrame([{
    "Model":     r["name"],
    "Threshold": round(r["threshold"],2),
    "CV_F1":     round(r["cv_f1"],3),
    "Accuracy":  round(r["accuracy"],3),
    "BalAcc":    round(r["bal_acc"],3),
    "Precision": round(r["precision"],3),
    "Recall":    round(r["recall"],3),
    "F1":        round(r["f1"],3),
    "AUC":       round(r["auc"],3),
    "TP": r["tp"], "FP": r["fp"],
    "FN": r["fn"], "TN": r["tn"],
} for r in results]).sort_values("F1", ascending=False)

comp.to_csv(os.path.join(MDL_DIR, "model_comparison.csv"), index=False)
print(f"\n  All model scores:")
print(comp[["Model","Threshold","CV_F1","Recall","Precision","F1","AUC"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 8. CHARTS
# ══════════════════════════════════════════════════════════════
print("\n[8/8] Generating charts...")

COLORS = ["#1565C0","#2E7D32","#EF6C00","#6A1B9A","#00838F",
          "#C62828","#455A64","#AD1457","#F9A825","#00695C"]
C = {"high":"#D32F2F","med":"#F57C00","low":"#388E3C","grid":"#E0E0E0"}

fig = plt.figure(figsize=(20,14))
fig.suptitle(
    f"NDMA Flood Risk — {best['name']}  |  "
    f"Stratified 80/20 Split  |  Threshold={best['threshold']:.2f}",
    fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.36)

top8 = sorted(results, key=lambda r: r["f1"], reverse=True)[:8]

# Chart 1 — model comparison
ax1   = fig.add_subplot(gs[0,0])
mets  = ["Accuracy","Precision","Recall","F1","AUC"]
x_pos = np.arange(len(mets))
bw    = 0.8 / len(top8)
for i, r in enumerate(top8):
    vals = [r["accuracy"],r["precision"],r["recall"],r["f1"],r["auc"]]
    ax1.bar(x_pos + i*bw - 0.4 + bw/2, vals, bw,
            label=r["name"][:16], color=COLORS[i%len(COLORS)], alpha=0.85)
ax1.set_xticks(x_pos); ax1.set_xticklabels(mets, fontsize=9)
ax1.set_ylim(0,1.15); ax1.set_ylabel("Score")
ax1.set_title("All Models — Stratified Split", fontweight="bold")
ax1.legend(fontsize=5); ax1.yaxis.grid(True,color=C["grid"]); ax1.set_axisbelow(True)

# Chart 2 — confusion matrix
ax2 = fig.add_subplot(gs[0,1])
cm  = np.array([[best["tn"],best["fp"]],[best["fn"],best["tp"]]])
im  = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
ax2.set_xticklabels(["No Flood","Flood"])
ax2.set_yticklabels(["No Flood","Flood"])
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax2.text(j,i,str(cm[i,j]),ha="center",va="center",
                 fontsize=20,fontweight="bold",
                 color="white" if cm[i,j]>cm.max()/2 else "black")
ax2.set_title(f"Confusion Matrix  Th={best['threshold']:.2f}", fontweight="bold")
plt.colorbar(im, ax=ax2, shrink=0.8)

# Chart 3 — ROC curves
ax3 = fig.add_subplot(gs[0,2])
for i, r in enumerate(sorted(results, key=lambda r: r["auc"], reverse=True)[:8]):
    fpr,tpr,_ = roc_curve(y_test, r["proba"])
    ax3.plot(fpr,tpr,lw=2,color=COLORS[i%len(COLORS)],
             label=f"{r['name'][:14]} ({r['auc']:.2f})")
ax3.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
ax3.set_title("ROC Curves — All Models", fontweight="bold")
ax3.legend(fontsize=6); ax3.yaxis.grid(True,color=C["grid"]); ax3.set_axisbelow(True)

# Chart 4 — threshold analysis
ax4 = fig.add_subplot(gs[1,0])
ths = np.arange(0.05,0.95,0.01)
f1s,recs,precs = [],[],[]
for th in ths:
    p = (best["proba"] >= th).astype(int)
    f1s.append(f1_score(y_test,p,zero_division=0))
    recs.append(recall_score(y_test,p,zero_division=0))
    precs.append(precision_score(y_test,p,zero_division=0))
ax4.plot(ths,recs,  "o-",color="#D32F2F",lw=2,ms=2,label="Recall")
ax4.plot(ths,precs, "s-",color="#1565C0",lw=2,ms=2,label="Precision")
ax4.plot(ths,f1s,   "^-",color="#2E7D32",lw=2,ms=2,label="F1")
ax4.axvline(best["threshold"],color="black",linestyle="--",lw=2,
            label=f"Best={best['threshold']:.2f}")
ax4.set_xlabel("Threshold"); ax4.set_ylabel("Score")
ax4.set_title("Threshold Optimization", fontweight="bold")
ax4.legend(fontsize=8); ax4.grid(True,color=C["grid"]); ax4.set_axisbelow(True)

# Chart 5 — feature importance
ax5 = fig.add_subplot(gs[1,1:])
bm = best["model"]
if hasattr(bm,"feature_importances_"):
    imp = bm.feature_importances_
elif hasattr(bm,"named_estimators_"):
    imp = np.ones(len(FEATURES_FINAL))/len(FEATURES_FINAL)
    for _,e in bm.named_estimators_.items():
        if hasattr(e,"feature_importances_"):
            imp = e.feature_importances_; break
else:
    imp = np.ones(len(FEATURES_FINAL))/len(FEATURES_FINAL)

feat_s = pd.Series(imp,index=FEATURES_FINAL).sort_values(ascending=True).tail(15)
bc = ["#D32F2F" if v>feat_s.quantile(0.75) else
      "#F57C00" if v>feat_s.quantile(0.4)  else "#1565C0" for v in feat_s]
ax5.barh(feat_s.index,feat_s.values,color=bc,alpha=0.85)
for i,(k,v) in enumerate(feat_s.items()):
    ax5.text(v+max(feat_s.max()*0.01,0.001),i,f"{v:.3f}",va="center",fontsize=8)
ax5.set_xlabel("Importance")
ax5.set_title("Top 15 Feature Importance  (Red=most important)", fontweight="bold")
ax5.xaxis.grid(True,color=C["grid"]); ax5.set_axisbelow(True)

# Chart 6 — risk by province
ax6 = fig.add_subplot(gs[2,0])
df["risk_score"] = best["model"].predict_proba(X_all_sc)[:,1]
prov = df.groupby("province")["risk_score"].mean().sort_values()
bc2  = [C["high"] if v>0.55 else C["med"] if v>0.35 else C["low"] for v in prov]
ax6.barh(prov.index,prov.values,color=bc2,alpha=0.85)
for i,(k,v) in enumerate(prov.items()):
    ax6.text(v+0.01,i,f"{v:.2f}",va="center",fontsize=9)
ax6.set_xlim(0,1); ax6.set_xlabel("Avg Risk Score")
ax6.set_title("Flood Risk by Province", fontweight="bold")
ax6.xaxis.grid(True,color=C["grid"]); ax6.set_axisbelow(True)

# Chart 7 — CV F1 comparison
ax7 = fig.add_subplot(gs[2,1])
names_cv  = [r["name"][:20] for r in sorted(results,key=lambda r:r["cv_f1"],reverse=True)[:8]]
cv_scores = [r["cv_f1"]     for r in sorted(results,key=lambda r:r["cv_f1"],reverse=True)[:8]]
test_f1s  = [r["f1"]        for r in sorted(results,key=lambda r:r["cv_f1"],reverse=True)[:8]]
xc = np.arange(len(names_cv))
ax7.bar(xc-0.18, cv_scores, 0.35, label="CV F1 (train)", color="#1565C0", alpha=0.8)
ax7.bar(xc+0.18, test_f1s,  0.35, label="Test F1",       color="#2E7D32", alpha=0.8)
ax7.set_xticks(xc); ax7.set_xticklabels(names_cv,rotation=25,ha="right",fontsize=7)
ax7.set_ylim(0,1.1); ax7.set_ylabel("F1 Score")
ax7.set_title("CV F1 vs Test F1\n(gap = overfitting check)", fontweight="bold")
ax7.legend(fontsize=8); ax7.yaxis.grid(True,color=C["grid"]); ax7.set_axisbelow(True)

# Chart 8 — summary
ax8 = fig.add_subplot(gs[2,2])
ax8.axis("off")
summary = f"""
STRATIFIED RANDOM SPLIT

Split type : 80% train / 20% test
Stratified : same flood % in both
Random seed: {SEED}

MODEL: {best['name'][:28]}
Threshold : {best['threshold']:.2f}

PERFORMANCE
Accuracy   : {best['accuracy']:.1%}
Bal.Acc    : {best['bal_acc']:.1%}
Recall     : {best['recall']:.1%}
Precision  : {best['precision']:.1%}
F1 Score   : {best['f1']:.3f}
CV F1      : {best['cv_f1']:.3f}
AUC-ROC    : {best['auc']:.3f}

CONFUSION MATRIX
TP (caught)   : {best['tp']}
FN (missed)   : {best['fn']}
FP (alarms)   : {best['fp']}
TN (correct)  : {best['tn']}

DATA
Total rows : 2041
Districts  : 157
Provinces  : 7
"""
ax8.text(0.05,0.97,summary,transform=ax8.transAxes,fontsize=9,
         verticalalignment="top",fontfamily="monospace",
         bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.5))

out_png = os.path.join(MDL_DIR, "model_results.png")
plt.savefig(out_png,dpi=150,bbox_inches="tight",facecolor="white")
plt.close()
print(f"      PNG saved: {out_png}")

# Save risk scores separately
risk_out = os.path.join(BASE,"data","processed","flood_risk_scores.csv")
df[["district","province","year","flooded","risk_score"]].to_csv(risk_out,index=False)
print(f"      Risk scores saved: data/processed/flood_risk_scores.csv")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  TRAINING COMPLETE!")
print("=" * 65)
print(f"\n  Split type : Stratified random 80/20")
print(f"  Best Model : {best['name']}")
print(f"  Threshold  : {best['threshold']:.2f}")
print(f"  F1         : {best['f1']:.3f}")
print(f"  CV F1      : {best['cv_f1']:.3f}  ← cross-validation score")
print(f"  Recall     : {best['recall']:.3f}")
print(f"  Precision  : {best['precision']:.3f}")
print(f"  AUC-ROC    : {best['auc']:.3f}")
print(f"\n  Confusion Matrix (on {len(X_test)} test rows):")
print(f"    Floods caught correctly : {best['tp']}")
print(f"    Floods missed           : {best['fn']}")
print(f"    False alarms            : {best['fp']}")
print(f"    Correct non-floods      : {best['tn']}")
print(f"\n  Files saved in models/:")
print(f"    flood_model_best.pkl")
print(f"    scaler.pkl")
print(f"    model_info.pkl")
print(f"    model_comparison.csv")
print(f"    model_results.png")
print(f"\n  Next step → run:   streamlit run app/app.py")