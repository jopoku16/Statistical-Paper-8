"""
Enhanced AKI Conformal Prediction Analysis
- Adds XGBoost and LightGBM
- Adds SHAP explanations
- External validation (train BIDMC, validate Emory)
- Improved risk stratification
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, accuracy_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time

OUT = r"C:\Users\19565\Desktop\Statistical Paper 8"
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")

# ================================================================
# STEP 1: Load and split by hospital
# ================================================================
print("=" * 65)
print("STEP 1: Load data and split by hospital")
print("=" * 65)

t0 = time.time()
df = pd.read_csv(os.path.join(OUT, "data", "physionet", "Dataset.csv"))
print(f"Loaded {len(df):,} records, {df['Patient_ID'].nunique():,} patients in {time.time()-t0:.1f}s")

# Hospital split
bidmc_ids = set(df.loc[df['Patient_ID'] <= 50000, 'Patient_ID'].unique())
emory_ids = set(df.loc[df['Patient_ID'] > 50000, 'Patient_ID'].unique())
print(f"BIDMC: {len(bidmc_ids):,} patients, Emory: {len(emory_ids):,} patients")

# ================================================================
# STEP 2: Construct AKI labels (KDIGO)
# ================================================================
print("\nConstructing AKI labels (KDIGO criteria)...")

cr_data = df.dropna(subset=['Creatinine']).groupby('Patient_ID')['Creatinine']
cr_first = cr_data.first()
cr_max = cr_data.max()
cr_count = cr_data.count()

aki_labels = pd.Series(0, index=cr_first.index)
fold_change = cr_max / cr_first
abs_change = cr_max - cr_first
aki_labels[(fold_change >= 1.5) | (abs_change >= 0.3)] = 1

print(f"AKI labels for {len(aki_labels):,} patients, prevalence: {aki_labels.mean():.1%}")

# ================================================================
# STEP 3: Feature engineering
# ================================================================
print("\nAggregating per-patient features...")

vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
labs = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
        'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
        'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium',
        'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI',
        'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

agg_dict = {}
for v in vitals:
    agg_dict[v] = ['mean', 'std', 'min', 'max']
for l in labs:
    agg_dict[l] = ['mean', 'std', 'min', 'max', 'first', 'last']
agg_dict['Age'] = 'first'
agg_dict['Gender'] = 'first'
agg_dict['Unit1'] = 'first'
agg_dict['Unit2'] = 'first'
agg_dict['HospAdmTime'] = 'first'
agg_dict['ICULOS'] = 'max'
agg_dict['SepsisLabel'] = 'max'

feat = df.groupby('Patient_ID').agg(agg_dict)
feat.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c
                 for c in feat.columns]
feat.rename(columns={'ICULOS_max': 'icu_los', 'SepsisLabel_max': 'ever_sepsis',
                      'Age_first': 'Age', 'Gender_first': 'Gender',
                      'Unit1_first': 'Unit1', 'Unit2_first': 'Unit2',
                      'HospAdmTime_first': 'HospAdmTime'}, inplace=True)

feat['cr_first'] = cr_first
feat['cr_count'] = cr_count
feat['shock_index'] = feat.get('HR_mean', 0) / feat.get('SBP_mean', 1).replace(0, np.nan)
feat['pulse_pressure'] = feat.get('SBP_mean', 0) - feat.get('DBP_mean', 0)
feat['bun_cr_ratio'] = feat.get('BUN_mean', 0) / feat['cr_first'].replace(0, np.nan)
feat['hr_variability'] = feat.get('HR_std', 0)

# Remove creatinine-derived leakage features
leak_cols = [c for c in feat.columns if 'Creatinine' in c]
feat.drop(columns=leak_cols, inplace=True, errors='ignore')

# Align with AKI labels
common = feat.index.intersection(aki_labels.index)
feat = feat.loc[common]
y_all = aki_labels.loc[common]

print(f"Final cohort: {len(feat):,} patients, AKI: {y_all.mean():.1%}")
print(f"Features: {feat.shape[1]}")

# ================================================================
# STEP 4: Split - BIDMC (train/cal) vs Emory (external test)
# ================================================================
print("\n" + "=" * 65)
print("STEP 4: Hospital-based split for external validation")
print("=" * 65)

bidmc_mask = feat.index.isin(bidmc_ids)
emory_mask = feat.index.isin(emory_ids)

X_bidmc = feat.loc[bidmc_mask]
y_bidmc = y_all.loc[bidmc_mask]
X_emory = feat.loc[emory_mask]
y_emory = y_all.loc[emory_mask]

print(f"BIDMC (development): {len(X_bidmc):,} (AKI={y_bidmc.sum():,}, {y_bidmc.mean():.1%})")
print(f"Emory (external):    {len(X_emory):,} (AKI={y_emory.sum():,}, {y_emory.mean():.1%})")

# Split BIDMC into train (70%) and calibration (30%)
from sklearn.model_selection import train_test_split
X_train, X_cal, y_train, y_cal = train_test_split(
    X_bidmc, y_bidmc, test_size=0.30, stratify=y_bidmc, random_state=42
)

print(f"  Train: {len(X_train):,} (AKI={y_train.sum():,}, {y_train.mean():.1%})")
print(f"  Cal:   {len(X_cal):,} (AKI={y_cal.sum():,}, {y_cal.mean():.1%})")
print(f"  External test (Emory): {len(X_emory):,}")

# Keep only numeric columns and drop all-NaN columns
num_cols = X_train.select_dtypes(include=[np.number]).columns
X_train = X_train[num_cols].dropna(axis=1, how='all')
kept_cols = X_train.columns
X_cal = X_cal[kept_cols]
X_emory = X_emory[kept_cols]
print(f"  Numeric features after dropping all-NaN: {len(kept_cols)}")

# Impute and scale
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=kept_cols, index=X_train.index)
X_cal_imp = pd.DataFrame(imputer.transform(X_cal),
                         columns=kept_cols, index=X_cal.index)
X_emory_imp = pd.DataFrame(imputer.transform(X_emory),
                           columns=kept_cols, index=X_emory.index)

X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp),
                          columns=kept_cols, index=X_train.index)
X_cal_sc = pd.DataFrame(scaler.transform(X_cal_imp),
                        columns=kept_cols, index=X_cal.index)
X_emory_sc = pd.DataFrame(scaler.transform(X_emory_imp),
                          columns=kept_cols, index=X_emory.index)

# ================================================================
# STEP 5: Train 5 models
# ================================================================
print("\n" + "=" * 65)
print("STEP 5: Train 5 ML models")
print("=" * 65)

models = {}

# Logistic Regression
t1 = time.time()
lr = LogisticRegression(C=0.1, max_iter=2000, solver='lbfgs', random_state=42)
lr.fit(X_train_sc, y_train)
models['Logistic Regression'] = ('lr', lr, X_cal_sc, X_emory_sc)
print(f"  Logistic Regression trained ({time.time()-t1:.1f}s)")

# Random Forest
t1 = time.time()
rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=20,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_imp, y_train)
models['Random Forest'] = ('rf', rf, X_cal_imp, X_emory_imp)
print(f"  Random Forest trained ({time.time()-t1:.1f}s)")

# Gradient Boosting
t1 = time.time()
gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, min_samples_leaf=20, random_state=42)
gb.fit(X_train_imp, y_train)
models['Gradient Boosting'] = ('gb', gb, X_cal_imp, X_emory_imp)
print(f"  Gradient Boosting trained ({time.time()-t1:.1f}s)")

# XGBoost
t1 = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, min_child_weight=20,
                               eval_metric='logloss', random_state=42,
                               n_jobs=-1, verbosity=0)
xgb_model.fit(X_train_imp, y_train)
models['XGBoost'] = ('xgb', xgb_model, X_cal_imp, X_emory_imp)
print(f"  XGBoost trained ({time.time()-t1:.1f}s)")

# LightGBM
t1 = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, min_child_samples=20,
                                random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train_imp, y_train)
models['LightGBM'] = ('lgb', lgb_model, X_cal_imp, X_emory_imp)
print(f"  LightGBM trained ({time.time()-t1:.1f}s)")

# Evaluate all models on Emory (external)
print("\nExternal validation (Emory):")
results = {}
for name, (tag, model, cal_X, ext_X) in models.items():
    if tag == 'lr':
        p_ext = model.predict_proba(X_emory_sc)[:, 1]
        p_cal = model.predict_proba(X_cal_sc)[:, 1]
    else:
        p_ext = model.predict_proba(X_emory_imp)[:, 1]
        p_cal = model.predict_proba(X_cal_imp)[:, 1]

    auc = roc_auc_score(y_emory, p_ext)
    auprc = average_precision_score(y_emory, p_ext)
    brier = brier_score_loss(y_emory, p_ext)
    pred = (p_ext >= 0.5).astype(int)
    f1 = f1_score(y_emory, pred)
    acc = accuracy_score(y_emory, pred)

    results[name] = {'AUC': auc, 'AUPRC': auprc, 'Brier': brier,
                     'F1': f1, 'Acc': acc, 'p_ext': p_ext, 'p_cal': p_cal}
    print(f"  {name:25s} AUC={auc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}  F1={f1:.3f}")

# Find best model
best_name = max(results, key=lambda k: results[k]['AUC'])
print(f"\nBest model: {best_name} (AUC={results[best_name]['AUC']:.3f})")

# ================================================================
# STEP 6: Calibration comparison (best model)
# ================================================================
print("\n" + "=" * 65)
print("STEP 6: Calibration comparison")
print("=" * 65)

p_cal_best = results[best_name]['p_cal']
p_ext_best = results[best_name]['p_ext']

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true.values[mask].mean() if hasattr(y_true, 'values') else y_true[mask].mean()
        total += mask.sum() * abs(avg_pred - avg_true)
    return total / len(y_true)

# Raw
raw_brier = brier_score_loss(y_emory, p_ext_best)
raw_ece = ece(y_emory, p_ext_best)

# Platt scaling
platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
platt.fit(p_cal_best.reshape(-1, 1), y_cal)
p_ext_platt = platt.predict_proba(p_ext_best.reshape(-1, 1))[:, 1]
platt_brier = brier_score_loss(y_emory, p_ext_platt)
platt_ece = ece(y_emory, p_ext_platt)

# Isotonic
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_cal_best, y_cal)
p_ext_iso = iso.predict(p_ext_best)
iso_brier = brier_score_loss(y_emory, p_ext_iso)
iso_ece = ece(y_emory, p_ext_iso)

print(f"  Raw        Brier={raw_brier:.4f}  ECE={raw_ece:.4f}")
print(f"  Platt      Brier={platt_brier:.4f}  ECE={platt_ece:.4f}")
print(f"  Isotonic   Brier={iso_brier:.4f}  ECE={iso_ece:.4f}")

# ================================================================
# STEP 7: Conformal prediction on external data
# ================================================================
print("\n" + "=" * 65)
print("STEP 7: Split conformal prediction (external validation)")
print("=" * 65)

alpha = 0.10
m = len(y_cal)

# Nonconformity scores on calibration set
s_cal = np.where(y_cal == 1, 1 - p_cal_best, p_cal_best)

q_level = np.ceil((1 - alpha) * (m + 1)) / m
q_hat = np.quantile(s_cal, min(q_level, 1.0))

print(f"Calibration size: m = {m:,}")
print(f"Conformal threshold q_hat = {q_hat:.4f}")

# Prediction sets on external test
p1_ext = p_ext_best
p0_ext = 1 - p1_ext

pred_sets = []
for i in range(len(y_emory)):
    s = set()
    if p0_ext[i] >= 1 - q_hat:
        s.add(0)
    if p1_ext[i] >= 1 - q_hat:
        s.add(1)
    pred_sets.append(s)

y_ext_arr = y_emory.values
covered = sum(y_ext_arr[i] in pred_sets[i] for i in range(len(y_ext_arr)))
singletons = sum(len(s) == 1 for s in pred_sets)
both = sum(len(s) == 2 for s in pred_sets)
empty = sum(len(s) == 0 for s in pred_sets)

cov_aki = sum(y_ext_arr[i] in pred_sets[i] for i in range(len(y_ext_arr)) if y_ext_arr[i] == 1)
n_aki = sum(y_ext_arr[i] == 1 for i in range(len(y_ext_arr)))
cov_noaki = sum(y_ext_arr[i] in pred_sets[i] for i in range(len(y_ext_arr)) if y_ext_arr[i] == 0)
n_noaki = sum(y_ext_arr[i] == 0 for i in range(len(y_ext_arr)))

print(f"\nMarginal conformal on external test:")
print(f"  Coverage: {covered/len(y_ext_arr):.1%} (target: {1-alpha:.0%})")
print(f"  Singletons: {singletons} ({singletons/len(y_ext_arr):.1%})")
print(f"  Both-class: {both} ({both/len(y_ext_arr):.1%})")
print(f"  Empty: {empty} ({empty/len(y_ext_arr):.1%})")
print(f"  Coverage (AKI=1): {cov_aki/n_aki:.1%}")
print(f"  Coverage (AKI=0): {cov_noaki/n_noaki:.1%}")

# ================================================================
# STEP 8: Class-conditional conformal (external)
# ================================================================
print("\n" + "=" * 65)
print("STEP 8: Class-conditional conformal prediction (external)")
print("=" * 65)

# Separate cal scores by class
s_cal_0 = 1 - p_cal_best[y_cal.values == 0]  # score for class 0 = 1 - p(0) = p(1)
s_cal_1 = 1 - p_cal_best[y_cal.values == 1]  # score for class 1 = 1 - p(1)

m0 = len(s_cal_0)
m1 = len(s_cal_1)

q0 = np.quantile(s_cal_0, min(np.ceil((1-alpha)*(m0+1))/m0, 1.0))
q1 = np.quantile(s_cal_1, min(np.ceil((1-alpha)*(m1+1))/m1, 1.0))

print(f"  Class 0: m={m0:,}, q_hat={q0:.4f}")
print(f"  Class 1: m={m1:,}, q_hat={q1:.4f}")

cc_sets = []
for i in range(len(y_ext_arr)):
    s = set()
    if (1 - p0_ext[i]) <= q0:  # p1 <= q0
        s.add(0)
    if (1 - p1_ext[i]) <= q1:  # 1-p1 <= q1
        s.add(1)
    cc_sets.append(s)

cc_covered = sum(y_ext_arr[i] in cc_sets[i] for i in range(len(y_ext_arr)))
cc_singletons = sum(len(s) == 1 for s in cc_sets)
cc_both = sum(len(s) == 2 for s in cc_sets)

cc_cov_aki = sum(y_ext_arr[i] in cc_sets[i] for i in range(len(y_ext_arr)) if y_ext_arr[i] == 1)
cc_cov_noaki = sum(y_ext_arr[i] in cc_sets[i] for i in range(len(y_ext_arr)) if y_ext_arr[i] == 0)

print(f"\nClass-conditional on external test:")
print(f"  Coverage (overall): {cc_covered/len(y_ext_arr):.1%}")
print(f"  Coverage (AKI=1): {cc_cov_aki/n_aki:.1%}")
print(f"  Coverage (AKI=0): {cc_cov_noaki/n_noaki:.1%}")
print(f"  Singletons: {cc_singletons/len(y_ext_arr):.1%}")
print(f"  Both-class: {cc_both/len(y_ext_arr):.1%}")

# ================================================================
# STEP 9: Mondrian conformal (external)
# ================================================================
print("\n" + "=" * 65)
print("STEP 9: Mondrian conformal prediction (external)")
print("=" * 65)

# By gender
for gname, gval in [('Female', 0.0), ('Male', 1.0)]:
    cal_mask_g = X_cal_imp['Gender'] == gval
    ext_mask_g = X_emory_imp['Gender'] == gval
    s_g = np.where(y_cal.values[cal_mask_g] == 1,
                   1 - p_cal_best[cal_mask_g],
                   p_cal_best[cal_mask_g])
    mg = len(s_g)
    qg = np.quantile(s_g, min(np.ceil((1-alpha)*(mg+1))/mg, 1.0))

    p1_g = p_ext_best[ext_mask_g]
    p0_g = 1 - p1_g
    y_g = y_ext_arr[ext_mask_g]

    sets_g = []
    for i in range(len(y_g)):
        s = set()
        if p0_g[i] >= 1 - qg:
            s.add(0)
        if p1_g[i] >= 1 - qg:
            s.add(1)
        sets_g.append(s)

    cov_g = sum(y_g[i] in sets_g[i] for i in range(len(y_g))) / len(y_g)
    avg_size = np.mean([len(s) for s in sets_g])
    print(f"  {gname}: n_cal={mg:,}, q={qg:.4f}, coverage={cov_g:.1%}, avg_size={avg_size:.2f}")

# By age group
for aname, lo, hi in [('<50', 0, 50), ('50-65', 50, 65), ('>65', 65, 200)]:
    cal_mask_a = (X_cal_imp['Age'] >= lo) & (X_cal_imp['Age'] < hi)
    ext_mask_a = (X_emory_imp['Age'] >= lo) & (X_emory_imp['Age'] < hi)
    s_a = np.where(y_cal.values[cal_mask_a] == 1,
                   1 - p_cal_best[cal_mask_a],
                   p_cal_best[cal_mask_a])
    ma = len(s_a)
    qa = np.quantile(s_a, min(np.ceil((1-alpha)*(ma+1))/ma, 1.0))

    p1_a = p_ext_best[ext_mask_a]
    p0_a = 1 - p1_a
    y_a = y_ext_arr[ext_mask_a]

    sets_a = []
    for i in range(len(y_a)):
        s = set()
        if p0_a[i] >= 1 - qa:
            s.add(0)
        if p1_a[i] >= 1 - qa:
            s.add(1)
        sets_a.append(s)

    cov_a = sum(y_a[i] in sets_a[i] for i in range(len(y_a))) / len(y_a)
    avg_size_a = np.mean([len(s) for s in sets_a])
    print(f"  Age {aname}: n_cal={ma:,}, q={qa:.4f}, coverage={cov_a:.1%}, avg_size={avg_size_a:.2f}")

# ================================================================
# STEP 10: Risk stratification (class-conditional, improved)
# ================================================================
print("\n" + "=" * 65)
print("STEP 10: Improved risk stratification (class-conditional)")
print("=" * 65)

risk = []
for i in range(len(y_ext_arr)):
    s = cc_sets[i]
    p = p1_ext[i]
    if s == {0}:
        risk.append('Low Risk')
    elif s == {1}:
        risk.append('High Risk')
    elif s == {0, 1}:
        risk.append('Uncertain')
    else:
        risk.append('Uncertain')

risk = np.array(risk)
for cat in ['Low Risk', 'Uncertain', 'High Risk']:
    mask = risk == cat
    n = mask.sum()
    if n == 0:
        continue
    aki_rate = y_ext_arr[mask].mean()
    mean_p = p1_ext[mask].mean()
    print(f"  {cat:12s}: n={n:5,} ({n/len(risk):5.1%}), AKI rate={aki_rate:.1%}, mean P(AKI)={mean_p:.3f}")

# ================================================================
# STEP 11: SHAP values
# ================================================================
print("\n" + "=" * 65)
print("STEP 11: SHAP explanations")
print("=" * 65)

best_tag = [v[0] for k, v in models.items() if k == best_name][0]
best_model_obj = [v[1] for k, v in models.items() if k == best_name][0]

if best_tag in ('gb', 'rf', 'xgb', 'lgb'):
    explainer = shap.TreeExplainer(best_model_obj)
    X_shap = X_emory_imp.sample(min(2000, len(X_emory_imp)), random_state=42)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:20]
    print("Top 20 SHAP features:")
    for i in top_idx:
        print(f"  {X_shap.columns[i]:30s} {mean_abs_shap[i]:.4f}")

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_shap.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    print("  Saved fig7_shap.png")

    # SHAP bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_shap, plot_type='bar', max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7b_shap_bar.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    print("  Saved fig7b_shap_bar.png")
else:
    print("  SHAP TreeExplainer not applicable for", best_name)

# ================================================================
# STEP 12: Cross-validation
# ================================================================
print("\n" + "=" * 65)
print("STEP 12: 5-fold cross-validation (BIDMC)")
print("=" * 65)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

X_bidmc_num = X_bidmc[kept_cols]

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_bidmc_num, y_bidmc), 1):
    X_tr = X_bidmc_num.iloc[tr_idx]
    y_tr = y_bidmc.iloc[tr_idx]
    X_te = X_bidmc_num.iloc[te_idx]
    y_te = y_bidmc.iloc[te_idx]

    imp_cv = SimpleImputer(strategy='median')
    X_tr_i = imp_cv.fit_transform(X_tr)
    X_te_i = imp_cv.transform(X_te)

    # Split test into cal/eval
    n_cal_cv = len(X_te_i) // 2
    X_cal_cv, X_eval_cv = X_te_i[:n_cal_cv], X_te_i[n_cal_cv:]
    y_cal_cv, y_eval_cv = y_te.values[:n_cal_cv], y_te.values[n_cal_cv:]

    gb_cv = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       min_samples_leaf=20, random_state=42)
    gb_cv.fit(X_tr_i, y_tr)

    p_eval = gb_cv.predict_proba(X_eval_cv)[:, 1]
    p_cal_cv = gb_cv.predict_proba(X_cal_cv)[:, 1]

    auc_cv = roc_auc_score(y_eval_cv, p_eval)

    # Marginal conformal
    s_cv = np.where(y_cal_cv == 1, 1 - p_cal_cv, p_cal_cv)
    mcv = len(s_cv)
    q_cv = np.quantile(s_cv, min(np.ceil((1-alpha)*(mcv+1))/mcv, 1.0))

    p1_ev = p_eval
    p0_ev = 1 - p1_ev
    sets_cv = []
    for i in range(len(y_eval_cv)):
        s = set()
        if p0_ev[i] >= 1 - q_cv:
            s.add(0)
        if p1_ev[i] >= 1 - q_cv:
            s.add(1)
        sets_cv.append(s)

    cov_cv = sum(y_eval_cv[i] in sets_cv[i] for i in range(len(y_eval_cv))) / len(y_eval_cv)

    # CC conformal
    s0_cv = p_cal_cv[y_cal_cv == 0]
    s1_cv = 1 - p_cal_cv[y_cal_cv == 1]
    m0_cv, m1_cv = len(s0_cv), len(s1_cv)
    q0_cv = np.quantile(s0_cv, min(np.ceil((1-alpha)*(m0_cv+1))/m0_cv, 1.0))
    q1_cv = np.quantile(s1_cv, min(np.ceil((1-alpha)*(m1_cv+1))/m1_cv, 1.0))

    cc_aki_cov = 0
    n_aki_cv = 0
    for i in range(len(y_eval_cv)):
        if y_eval_cv[i] == 1:
            n_aki_cv += 1
            if (1 - p1_ev[i]) <= q1_cv:
                cc_aki_cov += 1

    cc_aki_rate = cc_aki_cov / max(n_aki_cv, 1)

    cv_results.append({'fold': fold, 'auc': auc_cv, 'coverage': cov_cv, 'cc_aki_cov': cc_aki_rate})
    print(f"  Fold {fold}: AUC={auc_cv:.3f}, Coverage={cov_cv:.1%}, CC-AKI={cc_aki_rate:.1%}")

cv_auc = np.mean([r['auc'] for r in cv_results])
cv_auc_std = np.std([r['auc'] for r in cv_results])
cv_cov = np.mean([r['coverage'] for r in cv_results])
cv_cov_std = np.std([r['coverage'] for r in cv_results])
cv_cc = np.mean([r['cc_aki_cov'] for r in cv_results])
cv_cc_std = np.std([r['cc_aki_cov'] for r in cv_results])
print(f"\nCV Summary:")
print(f"  AUC: {cv_auc:.3f} +/- {cv_auc_std:.3f}")
print(f"  Coverage: {cv_cov:.1%} +/- {cv_cov_std:.1%}")
print(f"  CC-AKI Coverage: {cv_cc:.1%} +/- {cv_cc_std:.1%}")

# ================================================================
# STEP 13: Generate updated figures
# ================================================================
print("\n" + "=" * 65)
print("STEP 13: Generating figures")
print("=" * 65)

# Fig 1: ROC + PRC (all 5 models)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_emory, res['p_ext'])
    ax1.plot(fpr, tpr, color=colors[i], lw=2,
             label=f"{name} (AUC={res['AUC']:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('(a) ROC Curves - External Validation', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

for i, (name, res) in enumerate(results.items()):
    prec, rec, _ = precision_recall_curve(y_emory, res['p_ext'])
    ax2.plot(rec, prec, color=colors[i], lw=2,
             label=f"{name} (AUPRC={res['AUPRC']:.3f})")
ax2.axhline(y=y_emory.mean(), color='gray', ls='--', lw=1, label=f"Prevalence={y_emory.mean():.1%}")
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('(b) Precision-Recall Curves', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_roc_prc.png"), dpi=300)
plt.close()
print("  fig1_roc_prc.png (5 models)")

# Fig 2: Conformal comparison (marginal vs CC)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

categories = ['Singleton\n{0} or {1}', 'Both\n{0,1}', 'Empty']
marg_vals = [singletons/len(y_ext_arr)*100, both/len(y_ext_arr)*100, empty/len(y_ext_arr)*100]
cc_vals = [cc_singletons/len(y_ext_arr)*100, cc_both/len(y_ext_arr)*100, 0]

x = np.arange(len(categories))
w = 0.35
ax1.bar(x - w/2, marg_vals, w, label='Marginal', color='#1f77b4', alpha=0.8)
ax1.bar(x + w/2, cc_vals, w, label='Class-conditional', color='#2ca02c', alpha=0.8)
ax1.set_ylabel('Percentage of test set (%)', fontsize=11)
ax1.set_title('(a) Prediction Set Composition', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

classes = ['AKI = 0', 'AKI = 1', 'Overall']
marg_cov = [cov_noaki/n_noaki*100, cov_aki/n_aki*100, covered/len(y_ext_arr)*100]
cc_cov = [cc_cov_noaki/n_noaki*100, cc_cov_aki/n_aki*100, cc_covered/len(y_ext_arr)*100]

x2 = np.arange(len(classes))
ax2.bar(x2 - w/2, marg_cov, w, label='Marginal', color='#1f77b4', alpha=0.8)
ax2.bar(x2 + w/2, cc_cov, w, label='Class-conditional', color='#2ca02c', alpha=0.8)
ax2.axhline(y=90, color='red', ls='--', lw=2, label='90% target')
ax2.set_ylabel('Coverage (%)', fontsize=11)
ax2.set_title('(b) Coverage by Class', fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(classes, fontsize=10)
ax2.set_ylim(0, 105)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_conformal.png"), dpi=300)
plt.close()
print("  fig2_conformal.png")

# Fig 3: Feature importance (MDI + SHAP side by side)
fi = pd.read_csv(os.path.join(RES, "feature_importance.csv"))
# Use the new best model's feature importance
if best_tag in ('gb', 'xgb', 'lgb', 'rf'):
    fi_vals = best_model_obj.feature_importances_
    fi_new = pd.DataFrame({'feature': X_train_imp.columns, 'importance': fi_vals})
    fi_new = fi_new.sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(fi_new)-1, -1, -1), fi_new['importance'].values, color='#2ca02c', alpha=0.8)
    ax.set_yticks(range(len(fi_new)-1, -1, -1))
    ax.set_yticklabels(fi_new['feature'].values, fontsize=10)
    ax.set_xlabel('Feature Importance (MDI)', fontsize=12)
    ax.set_title(f'Top 15 Features ({best_name})', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_importance.png"), dpi=300)
    plt.close()
    print("  fig3_importance.png")

# Fig 4: Decision curve analysis
thresholds = np.linspace(0.01, 0.99, 200)
prevalence = y_emory.mean()

net_benefit_model = []
net_benefit_all = []
for t in thresholds:
    tp = ((p_ext_best >= t) & (y_emory == 1)).sum()
    fp = ((p_ext_best >= t) & (y_emory == 0)).sum()
    n = len(y_emory)
    nb = tp/n - fp/n * (t / (1 - t))
    net_benefit_model.append(nb)
    nb_all = prevalence - (1 - prevalence) * t / (1 - t)
    net_benefit_all.append(max(nb_all, 0))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(thresholds, net_benefit_model, 'b-', lw=2, label=best_name)
ax.plot(thresholds, net_benefit_all, 'r--', lw=1.5, label='Treat All')
ax.axhline(y=0, color='gray', ls='-', lw=1, label='Treat None')
ax.set_xlabel('Threshold Probability', fontsize=12)
ax.set_ylabel('Net Benefit', fontsize=12)
ax.set_title('Decision Curve Analysis (External Validation)', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 0.8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_decision_curve.png"), dpi=300)
plt.close()
print("  fig4_decision_curve.png")

# Fig 6: CV stability
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
folds = [r['fold'] for r in cv_results]

axes[0].bar(folds, [r['auc'] for r in cv_results], color='#1f77b4', alpha=0.8)
axes[0].axhline(y=cv_auc, color='red', ls='--', lw=2, label=f'Mean={cv_auc:.3f}')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('AUC')
axes[0].set_title('(a) AUC across folds')
axes[0].set_ylim(0.9, 0.96)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(folds, [r['coverage']*100 for r in cv_results], color='#2ca02c', alpha=0.8)
axes[1].axhline(y=90, color='red', ls='--', lw=2, label='90% target')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Coverage (%)')
axes[1].set_title('(b) Marginal Coverage')
axes[1].set_ylim(85, 95)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].bar(folds, [r['cc_aki_cov']*100 for r in cv_results], color='#ff7f0e', alpha=0.8)
axes[2].axhline(y=90, color='red', ls='--', lw=2, label='90% target')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('Coverage (%)')
axes[2].set_title('(c) CC-AKI Coverage')
axes[2].set_ylim(80, 100)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_cv_stability.png"), dpi=300)
plt.close()
print("  fig6_cv_stability.png")

# Fig 8: Calibration
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax_i, (pvals, title) in enumerate([
    (p_ext_best, 'Raw'), (p_ext_platt, 'Platt'), (p_ext_iso, 'Isotonic')
]):
    bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_true = []
    for j in range(10):
        mask = (pvals >= bins[j]) & (pvals < bins[j+1])
        if mask.sum() > 0:
            bin_centers.append(pvals[mask].mean())
            bin_true.append(y_emory.values[mask].mean())
    axes[ax_i].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[ax_i].scatter(bin_centers, bin_true, s=80, zorder=5, color='#d62728')
    axes[ax_i].plot(bin_centers, bin_true, lw=2, color='#d62728')
    axes[ax_i].set_xlabel('Predicted Probability')
    axes[ax_i].set_ylabel('Observed Frequency')
    axes[ax_i].set_title(f'{title}')
    axes[ax_i].set_xlim(0, 1)
    axes[ax_i].set_ylim(0, 1)
    axes[ax_i].grid(True, alpha=0.3)

plt.suptitle('Calibration Comparison (External Validation)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig8_calibration.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  fig8_calibration.png")

# ================================================================
# STEP 14: Save updated results
# ================================================================
print("\n" + "=" * 65)
print("STEP 14: Saving results")
print("=" * 65)

# Model comparison
mc = pd.DataFrame(results).T[['AUC', 'AUPRC', 'Brier', 'F1', 'Acc']]
mc.to_csv(os.path.join(RES, "model_comparison.csv"))

# Summary
summary = {
    'dataset': 'PhysioNet 2019 (BIDMC train, Emory external)',
    'n_bidmc': len(X_bidmc),
    'n_emory': len(X_emory),
    'n_train': len(X_train),
    'n_cal': len(X_cal),
    'aki_prev_bidmc': float(y_bidmc.mean()),
    'aki_prev_emory': float(y_emory.mean()),
    'n_features': int(feat.shape[1]),
    'best_model': best_name,
    'best_auc_external': float(results[best_name]['AUC']),
    'best_auprc_external': float(results[best_name]['AUPRC']),
    'marginal_coverage_ext': float(covered / len(y_ext_arr)),
    'marginal_cov_aki_ext': float(cov_aki / n_aki),
    'marginal_cov_noaki_ext': float(cov_noaki / n_noaki),
    'cc_coverage_ext': float(cc_covered / len(y_ext_arr)),
    'cc_cov_aki_ext': float(cc_cov_aki / n_aki),
    'cc_cov_noaki_ext': float(cc_cov_noaki / n_noaki),
    'raw_ece': float(raw_ece),
    'iso_ece': float(iso_ece),
    'cv_auc_mean': float(cv_auc),
    'cv_auc_std': float(cv_auc_std),
    'cv_cov_mean': float(cv_cov),
    'cv_cov_std': float(cv_cov_std),
    'cv_cc_aki_mean': float(cv_cc),
    'cv_cc_aki_std': float(cv_cc_std),
    'n_models': 5,
}
with open(os.path.join(RES, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print("  Results saved.")

# ================================================================
print("\n" + "=" * 65)
print("ENHANCED ANALYSIS COMPLETE")
print("=" * 65)
print(f"  Development: {len(X_bidmc):,} BIDMC patients")
print(f"  External:    {len(X_emory):,} Emory patients")
print(f"  Models: 5 (LR, RF, GB, XGBoost, LightGBM)")
print(f"  Best: {best_name} (ext AUC={results[best_name]['AUC']:.3f})")
print(f"  Marginal coverage (ext): {covered/len(y_ext_arr):.1%}")
print(f"  CC coverage: AKI={cc_cov_aki/n_aki:.1%}, NoAKI={cc_cov_noaki/n_noaki:.1%}")
print(f"  SHAP: computed")
print(f"  CV: AUC={cv_auc:.3f}+/-{cv_auc_std:.3f}")
