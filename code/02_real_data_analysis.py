"""
AKI Prediction with Conformal Prediction -- Real PhysioNet 2019 Data
Statistical Paper 8 (Rebuilt)
40,336 real ICU patients from two hospital systems
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json, time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG  = ROOT / "figures"
RES  = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

np.random.seed(42)

# ================================================================
# STEP 1: LOAD & AGGREGATE REAL CLINICAL DATA
# ================================================================
print("=" * 65)
print("STEP 1: Load PhysioNet 2019 data (40,336 ICU patients)")
print("=" * 65)

t0 = time.time()
raw = pd.read_csv(DATA / "physionet" / "Dataset.csv")
print(f"Loaded {len(raw):,} hourly records, {raw['Patient_ID'].nunique():,} patients in {time.time()-t0:.1f}s")

vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
labs = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
        'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
        'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
        'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
        'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
        'Platelets']
demographics = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']

# --- Construct AKI label using KDIGO criteria ---
print("\nConstructing AKI labels (KDIGO criteria)...")

def compute_aki_label(group):
    cr = group['Creatinine'].dropna()
    if len(cr) < 1:
        return np.nan
    baseline = cr.iloc[0]
    if np.isnan(baseline):
        baseline = cr.dropna().iloc[0] if len(cr.dropna()) > 0 else np.nan
    if np.isnan(baseline):
        return np.nan
    peak = cr.max()
    # KDIGO Stage 1+: >= 1.5x baseline OR absolute increase >= 0.3
    if peak >= 1.5 * baseline or (peak - baseline) >= 0.3:
        return 1
    return 0

aki_labels = raw.groupby('Patient_ID').apply(compute_aki_label)
aki_labels.name = 'aki_label'
print(f"AKI labels computed for {aki_labels.notna().sum():,} patients")
print(f"AKI prevalence (among labeled): {aki_labels.dropna().mean():.1%}")

# --- Aggregate features per patient ---
print("\nAggregating per-patient features...")

agg_dict = {}
for v in vitals:
    agg_dict[v] = ['mean', 'std', 'min', 'max']
for l in labs:
    agg_dict[l] = ['mean', 'std', 'min', 'max', 'first', 'last']

patient_features = raw.groupby('Patient_ID').agg(agg_dict)
patient_features.columns = ['_'.join(col) for col in patient_features.columns]

# Add demographics (constant per patient)
demo = raw.groupby('Patient_ID')[demographics].first()
patient_features = patient_features.join(demo)

# Add ICU length of stay
icu_los = raw.groupby('Patient_ID')['ICULOS'].max()
icu_los.name = 'icu_los'
patient_features = patient_features.join(icu_los)

# Add sepsis label (ever septic during stay)
sepsis = raw.groupby('Patient_ID')['SepsisLabel'].max()
sepsis.name = 'ever_sepsis'
patient_features = patient_features.join(sepsis)

# Add creatinine trajectory features
cr_traj = raw.groupby('Patient_ID')['Creatinine'].agg(
    cr_count='count',
    cr_first=lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan,
    cr_last=lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan,
    cr_peak='max',
    cr_delta=lambda x: (x.dropna().iloc[-1] - x.dropna().iloc[0])
        if len(x.dropna()) >= 2 else np.nan
)
patient_features = patient_features.join(cr_traj)

# Join AKI label
df = patient_features.join(aki_labels)

# Drop patients with no AKI label
df = df.dropna(subset=['aki_label']).copy()
df['aki_label'] = df['aki_label'].astype(int)

print(f"\nFinal cohort: {len(df):,} patients")
print(f"AKI prevalence: {df['aki_label'].mean():.1%} ({df['aki_label'].sum():,}/{len(df):,})")
print(f"Features: {df.shape[1] - 1}")

# ================================================================
# STEP 2: FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 65)
print("STEP 2: Feature engineering")
print("=" * 65)

# Remove outcome-leaking features
drop_cols = ['aki_label',
    'Creatinine_last', 'Creatinine_max',
    'cr_last', 'cr_peak', 'cr_delta',
    'Creatinine_mean', 'Creatinine_std', 'Creatinine_min',
    'Creatinine_first',
]

feature_cols = [c for c in df.columns if c not in drop_cols]

# Add derived features
df['shock_index'] = df['HR_mean'] / df['SBP_mean'].clip(lower=60)
df['pulse_pressure'] = df['SBP_mean'] - df['DBP_mean']
df['bun_cr_ratio'] = df['BUN_mean'] / df['cr_first'].clip(lower=0.1)
df['hr_variability'] = df['HR_std'] / df['HR_mean'].clip(lower=40)

derived_feats = ['shock_index', 'pulse_pressure', 'bun_cr_ratio', 'hr_variability']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df['aki_label'].values

print(f"Features after engineering: {X.shape[1]}")
print(f"Missingness range: {X.isnull().mean().min():.1%} - {X.isnull().mean().max():.1%}")

# Impute
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
feature_names = list(X_imp.columns)

# ================================================================
# STEP 3: TRAIN / CALIBRATION / TEST SPLIT
# ================================================================
print("\n" + "=" * 65)
print("STEP 3: Data splitting (60/20/20)")
print("=" * 65)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_imp, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"Train:       {len(X_train):,} (AKI={y_train.sum():,}, {y_train.mean():.1%})")
print(f"Calibration: {len(X_cal):,} (AKI={y_cal.sum():,}, {y_cal.mean():.1%})")
print(f"Test:        {len(X_test):,} (AKI={y_test.sum():,}, {y_test.mean():.1%})")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_cal_sc = scaler.transform(X_cal)
X_test_sc = scaler.transform(X_test)

# ================================================================
# STEP 4: TRAIN 5 ML MODELS
# ================================================================
print("\n" + "=" * 65)
print("STEP 4: Train ML models (5 classifiers)")
print("=" * 65)

models = {}

# 1. Logistic Regression
models['Logistic Regression'] = {
    'model': LogisticRegression(max_iter=2000, C=0.1, penalty='l2',
                                solver='lbfgs', random_state=42),
    'use_scaled': True
}

# 2. Random Forest
models['Random Forest'] = {
    'model': RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'use_scaled': False
}

# 3. Gradient Boosting
models['Gradient Boosting'] = {
    'model': GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42),
    'use_scaled': False
}

# 4. XGBoost
try:
    from xgboost import XGBClassifier
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models['XGBoost'] = {
        'model': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, eval_metric='logloss',
            random_state=42, n_jobs=-1),
        'use_scaled': False
    }
except ImportError:
    print("  XGBoost not available")

# 5. LightGBM
try:
    from lightgbm import LGBMClassifier
    models['LightGBM'] = {
        'model': LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            class_weight='balanced', random_state=42, n_jobs=-1,
            verbose=-1),
        'use_scaled': False
    }
except ImportError:
    print("  LightGBM not available")

results = {}
trained = {}

for name, cfg in models.items():
    t1 = time.time()
    mdl = cfg['model']
    scaled = cfg['use_scaled']

    Xtr = X_train_sc if scaled else X_train.values
    Xca = X_cal_sc if scaled else X_cal.values
    Xte = X_test_sc if scaled else X_test.values

    mdl.fit(Xtr, y_train)
    p_cal = mdl.predict_proba(Xca)[:, 1]
    p_test = mdl.predict_proba(Xte)[:, 1]

    auc = roc_auc_score(y_test, p_test)
    ap = average_precision_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)
    pred = (p_test >= 0.5).astype(int)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)

    results[name] = {
        'AUC': auc, 'AUPRC': ap, 'Brier': brier,
        'F1': f1, 'Accuracy': acc,
        'prob_cal': p_cal, 'prob_test': p_test
    }
    trained[name] = {'model': mdl, 'scaled': scaled}
    print(f"  {name:25s} AUC={auc:.3f} AUPRC={ap:.3f} "
          f"Brier={brier:.3f} F1={f1:.3f} ({time.time()-t1:.1f}s)")

best_name = max(results, key=lambda k: results[k]['AUC'])
print(f"\nBest model: {best_name} (AUC={results[best_name]['AUC']:.3f})")

# ================================================================
# STEP 5: CALIBRATION COMPARISON
# ================================================================
print("\n" + "=" * 65)
print("STEP 5: Calibration comparison (raw vs Platt vs isotonic)")
print("=" * 65)

best_mdl = trained[best_name]['model']
best_scaled = trained[best_name]['scaled']
Xtr_b = X_train_sc if best_scaled else X_train.values
Xte_b = X_test_sc if best_scaled else X_test.values
Xca_b = X_cal_sc if best_scaled else X_cal.values

p_raw = results[best_name]['prob_test']
p_cal_raw = results[best_name]['prob_cal']

# Platt scaling (manual sigmoid fit on calibration set)
from sklearn.linear_model import LogisticRegression as LR_cal
platt_lr = LR_cal(max_iter=1000)
platt_lr.fit(p_cal_raw.reshape(-1, 1), y_cal)
p_platt = platt_lr.predict_proba(p_raw.reshape(-1, 1))[:, 1]

# Isotonic regression
from sklearn.isotonic import IsotonicRegression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(p_cal_raw, y_cal)
p_iso = iso_reg.predict(p_raw)

for label, probs in [('Raw', p_raw), ('Platt', p_platt), ('Isotonic', p_iso)]:
    brier_c = brier_score_loss(y_test, probs)
    ece_bins = 10
    bin_edges = np.linspace(0, 1, ece_bins + 1)
    ece = 0
    for i in range(ece_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_test[mask].mean() - probs[mask].mean())
    ece /= len(y_test)
    print(f"  {label:10s} Brier={brier_c:.4f}  ECE={ece:.4f}")

# ================================================================
# STEP 6: SPLIT CONFORMAL PREDICTION
# ================================================================
print("\n" + "=" * 65)
print("STEP 6: Split conformal prediction")
print("=" * 65)

alpha = 0.10
best_prob_cal = results[best_name]['prob_cal']
best_prob_test = results[best_name]['prob_test']
m = len(y_cal)

scores_cal = np.where(y_cal == 1, 1 - best_prob_cal, best_prob_cal)

q_level = np.ceil((1 - alpha) * (m + 1)) / m
q_hat = np.quantile(scores_cal, q_level, method='higher')

print(f"Calibration size: m = {m:,}")
print(f"Conformal threshold q_hat = {q_hat:.4f}")
print(f"Coverage bound: [{1-alpha:.2f}, {1-alpha + 1/(m+1):.6f}]")

# Generate prediction sets
prediction_sets = []
for p in best_prob_test:
    ps = set()
    if (1 - p) <= q_hat:
        ps.add(1)
    if p <= q_hat:
        ps.add(0)
    prediction_sets.append(ps)

covered = sum(1 for i, ps in enumerate(prediction_sets) if y_test[i] in ps)
coverage = covered / len(y_test)

singleton = sum(1 for ps in prediction_sets if len(ps) == 1)
both = sum(1 for ps in prediction_sets if len(ps) == 2)
empty = sum(1 for ps in prediction_sets if len(ps) == 0)

aki_mask = y_test == 1
cov_aki = sum(1 for i in range(len(y_test))
              if aki_mask[i] and y_test[i] in prediction_sets[i]) / max(aki_mask.sum(), 1)
cov_noaki = sum(1 for i in range(len(y_test))
                if not aki_mask[i] and y_test[i] in prediction_sets[i]) / max((~aki_mask).sum(), 1)

print(f"\nTest results:")
print(f"  Coverage: {coverage:.1%} (target: {1-alpha:.0%})")
print(f"  Singletons: {singleton} ({singleton/len(y_test):.1%})")
print(f"  Both-class: {both} ({both/len(y_test):.1%})")
print(f"  Empty: {empty} ({empty/len(y_test):.1%})")
print(f"  Coverage (AKI=1): {cov_aki:.1%}")
print(f"  Coverage (AKI=0): {cov_noaki:.1%}")

# ================================================================
# STEP 7: CLASS-CONDITIONAL CONFORMAL (Romano et al. 2020)
# ================================================================
print("\n" + "=" * 65)
print("STEP 7: Class-conditional conformal prediction")
print("=" * 65)

cc_results = {}
for cls in [0, 1]:
    cls_mask = y_cal == cls
    cls_scores = scores_cal[cls_mask]
    m_c = len(cls_scores)
    q_c = np.quantile(cls_scores,
                      np.ceil((1 - alpha) * (m_c + 1)) / m_c,
                      method='higher')
    cc_results[cls] = {'m': m_c, 'q': q_c}
    print(f"  Class {cls}: m={m_c:,}, q_hat={q_c:.4f}")

# Generate class-conditional prediction sets
cc_prediction_sets = []
for p in best_prob_test:
    ps = set()
    if (1 - p) <= cc_results[1]['q']:
        ps.add(1)
    if p <= cc_results[0]['q']:
        ps.add(0)
    cc_prediction_sets.append(ps)

cc_covered = sum(1 for i, ps in enumerate(cc_prediction_sets) if y_test[i] in ps)
cc_coverage = cc_covered / len(y_test)

cc_cov_aki = sum(1 for i in range(len(y_test))
                 if aki_mask[i] and y_test[i] in cc_prediction_sets[i]) / max(aki_mask.sum(), 1)
cc_cov_noaki = sum(1 for i in range(len(y_test))
                   if not aki_mask[i] and y_test[i] in cc_prediction_sets[i]) / max((~aki_mask).sum(), 1)
cc_singleton = sum(1 for ps in cc_prediction_sets if len(ps) == 1)
cc_both = sum(1 for ps in cc_prediction_sets if len(ps) == 2)

print(f"\nClass-conditional results:")
print(f"  Coverage (overall): {cc_coverage:.1%}")
print(f"  Coverage (AKI=1): {cc_cov_aki:.1%}")
print(f"  Coverage (AKI=0): {cc_cov_noaki:.1%}")
print(f"  Singletons: {cc_singleton/len(y_test):.1%}")
print(f"  Both-class: {cc_both/len(y_test):.1%}")

# ================================================================
# STEP 8: MONDRIAN CONFORMAL
# ================================================================
print("\n" + "=" * 65)
print("STEP 8: Mondrian conformal prediction")
print("=" * 65)

# Group by age and gender
cal_info = pd.DataFrame({
    'prob': best_prob_cal, 'y': y_cal, 'score': scores_cal,
    'age': X_cal['Age'].values,
    'gender': X_cal['Gender'].values,
})
cal_info['age_group'] = pd.cut(cal_info['age'],
    bins=[0, 50, 65, 120], labels=['<50', '50-65', '>65'])

test_info = pd.DataFrame({
    'prob': best_prob_test, 'y': y_test,
    'age': X_test['Age'].values,
    'gender': X_test['Gender'].values,
})
test_info['age_group'] = pd.cut(test_info['age'],
    bins=[0, 50, 65, 120], labels=['<50', '50-65', '>65'])

mondrian_results = {}
for group_col in ['gender', 'age_group']:
    print(f"\n  Mondrian by {group_col}:")
    for g in sorted(cal_info[group_col].dropna().unique(), key=str):
        cal_m = cal_info[group_col] == g
        test_m = test_info[group_col] == g
        if cal_m.sum() < 10 or test_m.sum() < 5:
            continue
        g_scores = cal_info.loc[cal_m, 'score'].values
        m_g = len(g_scores)
        q_g = np.quantile(g_scores,
            np.ceil((1-alpha)*(m_g+1))/m_g, method='higher')
        g_probs = test_info.loc[test_m, 'prob'].values
        g_y = test_info.loc[test_m, 'y'].values
        g_psets = []
        for p in g_probs:
            ps = set()
            if (1-p) <= q_g: ps.add(1)
            if p <= q_g: ps.add(0)
            g_psets.append(ps)
        g_cov = sum(1 for i,ps in enumerate(g_psets) if g_y[i] in ps)/len(g_y)
        g_sz = np.mean([len(ps) for ps in g_psets])
        key = f"{group_col}={g}"
        mondrian_results[key] = {
            'n_cal': m_g, 'n_test': len(g_y),
            'q_hat': float(q_g), 'coverage': float(g_cov),
            'avg_set_size': float(g_sz)
        }
        print(f"    {key}: n_cal={m_g:,}, coverage={g_cov:.1%}, "
              f"avg_size={g_sz:.2f}")

# ================================================================
# STEP 9: REAL P-VALUES FOR COHORT TABLE
# ================================================================
print("\n" + "=" * 65)
print("STEP 9: Compute real p-values for cohort table")
print("=" * 65)

aki_df = df[df['aki_label'] == 1]
noaki_df = df[df['aki_label'] == 0]

cohort_stats = {}
continuous_vars = {
    'Age': 'Age',
    'HR_mean': 'Heart Rate',
    'SBP_mean': 'Systolic BP',
    'MAP_mean': 'Mean Arterial Pressure',
    'Resp_mean': 'Respiratory Rate',
    'O2Sat_mean': 'SpO2',
    'Temp_mean': 'Temperature',
    'BUN_mean': 'BUN',
    'cr_first': 'Baseline Creatinine',
    'Lactate_mean': 'Lactate',
    'WBC_mean': 'WBC',
    'Platelets_mean': 'Platelets',
    'Hgb_mean': 'Hemoglobin',
    'Glucose_mean': 'Glucose',
    'icu_los': 'ICU LOS (hours)',
}

print(f"{'Variable':30s} {'No AKI':>25s} {'AKI':>25s} {'p-value':>12s}")
print("-" * 95)
for col, label in continuous_vars.items():
    a = noaki_df[col].dropna()
    b = aki_df[col].dropna()
    if len(a) < 5 or len(b) < 5:
        continue
    stat, pval = stats.mannwhitneyu(a, b, alternative='two-sided')
    med_a = f"{a.median():.1f} [{a.quantile(0.25):.1f}-{a.quantile(0.75):.1f}]"
    med_b = f"{b.median():.1f} [{b.quantile(0.25):.1f}-{b.quantile(0.75):.1f}]"
    pstr = f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}"
    cohort_stats[label] = {
        'noaki': med_a, 'aki': med_b, 'p': pval, 'pstr': pstr
    }
    print(f"  {label:28s} {med_a:>25s} {med_b:>25s} {pstr:>12s}")

# Categorical: Gender
g_a = (noaki_df['Gender'] == 1).mean() * 100
g_b = (aki_df['Gender'] == 1).mean() * 100
ct = pd.crosstab(df['aki_label'], df['Gender'])
chi2, p_gender, _, _ = stats.chi2_contingency(ct)
pstr_g = f"{p_gender:.1e}" if p_gender < 0.001 else f"{p_gender:.3f}"
cohort_stats['Male (%)'] = {'noaki': f"{g_a:.1f}", 'aki': f"{g_b:.1f}",
                             'p': p_gender, 'pstr': pstr_g}
print(f"  {'Male (%)':28s} {g_a:>25.1f} {g_b:>25.1f} {pstr_g:>12s}")

# Sepsis
s_a = noaki_df['ever_sepsis'].mean() * 100
s_b = aki_df['ever_sepsis'].mean() * 100
ct2 = pd.crosstab(df['aki_label'], df['ever_sepsis'])
chi2_s, p_sepsis, _, _ = stats.chi2_contingency(ct2)
pstr_s = f"{p_sepsis:.1e}" if p_sepsis < 0.001 else f"{p_sepsis:.3f}"
cohort_stats['Sepsis (%)'] = {'noaki': f"{s_a:.1f}", 'aki': f"{s_b:.1f}",
                               'p': p_sepsis, 'pstr': pstr_s}
print(f"  {'Sepsis (%)':28s} {s_a:>25.1f} {s_b:>25.1f} {pstr_s:>12s}")

# ================================================================
# STEP 10: FEATURE IMPORTANCE + SHAP
# ================================================================
print("\n" + "=" * 65)
print("STEP 10: Feature importance + SHAP")
print("=" * 65)

best_mdl_obj = trained[best_name]['model']
if hasattr(best_mdl_obj, 'feature_importances_'):
    importances = best_mdl_obj.feature_importances_
else:
    importances = np.abs(best_mdl_obj.coef_[0])

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
feat_imp.to_csv(RES / "feature_importance.csv", index=False)
print("Top 15 features:")
for _, row in feat_imp.head(15).iterrows():
    print(f"  {row['feature']:35s} {row['importance']:.4f}")

# SHAP
try:
    import shap
    print("\nComputing SHAP values...")
    Xte_shap = X_test.values if not trained[best_name]['scaled'] else X_test_sc
    if best_name in ['Gradient Boosting', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(best_mdl_obj)
    else:
        bg = shap.sample(pd.DataFrame(Xtr_b, columns=feature_names), 200)
        explainer = shap.KernelExplainer(best_mdl_obj.predict_proba, bg)

    # Use subset for speed
    n_shap = min(2000, len(Xte_shap))
    shap_idx = np.random.choice(len(Xte_shap), n_shap, replace=False)
    shap_values = explainer.shap_values(
        pd.DataFrame(Xte_shap[shap_idx], columns=feature_names)
        if not trained[best_name]['scaled']
        else Xte_shap[shap_idx]
    )

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals,
        pd.DataFrame(Xte_shap[shap_idx], columns=feature_names),
        max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig7_shap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig7_shap.png saved")

    # Mean absolute SHAP
    mean_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    mean_shap.to_csv(RES / "shap_importance.csv", index=False)
    has_shap = True
except ImportError:
    print("  SHAP not available, skipping")
    has_shap = False

# ================================================================
# STEP 11: CROSS-VALIDATION
# ================================================================
print("\n" + "=" * 65)
print("STEP 11: 5-fold cross-validation")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
cv_coverages = []
cv_cc_coverages_aki = []

X_full = X_imp.values
y_full = y

for fold, (train_idx, test_idx) in enumerate(cv.split(X_full, y_full)):
    X_tr, X_te = X_full[train_idx], X_full[test_idx]
    y_tr, y_te = y_full[train_idx], y_full[test_idx]

    X_tr2, X_ca2, y_tr2, y_ca2 = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=fold, stratify=y_tr
    )

    if trained[best_name]['scaled']:
        sc2 = StandardScaler()
        X_tr2 = sc2.fit_transform(X_tr2)
        X_ca2 = sc2.transform(X_ca2)
        X_te = sc2.transform(X_te)

    mdl_cv = best_mdl_obj.__class__(**best_mdl_obj.get_params())
    mdl_cv.fit(X_tr2, y_tr2)
    p_ca = mdl_cv.predict_proba(X_ca2)[:, 1]
    p_te = mdl_cv.predict_proba(X_te)[:, 1]

    cv_aucs.append(roc_auc_score(y_te, p_te))

    # Marginal conformal
    sc_ca = np.where(y_ca2 == 1, 1 - p_ca, p_ca)
    m_f = len(sc_ca)
    q_f = np.quantile(sc_ca, np.ceil((1-alpha)*(m_f+1))/m_f, method='higher')
    cov_f = sum(1 for i,p in enumerate(p_te)
                if y_te[i] in ({1} if (1-p)<=q_f else set()) |
                              ({0} if p<=q_f else set())) / len(y_te)
    cv_coverages.append(cov_f)

    # Class-conditional
    for cls in [0, 1]:
        cls_m = y_ca2 == cls
        cls_sc = sc_ca[cls_m]
        m_c = len(cls_sc)
        cc_results[cls] = np.quantile(cls_sc,
            np.ceil((1-alpha)*(m_c+1))/m_c, method='higher')
    cc_cov_f = sum(1 for i,p in enumerate(p_te)
                   if y_te[i] == 1 and (1-p) <= cc_results[1]) / max((y_te==1).sum(),1)
    cv_cc_coverages_aki.append(cc_cov_f)

    print(f"  Fold {fold+1}: AUC={cv_aucs[-1]:.3f}, "
          f"Coverage={cv_coverages[-1]:.1%}, "
          f"CC-AKI-Coverage={cv_cc_coverages_aki[-1]:.1%}")

print(f"\nCV Summary:")
print(f"  AUC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")
print(f"  Coverage: {np.mean(cv_coverages):.1%} +/- {np.std(cv_coverages)*100:.1f}%")
print(f"  CC-AKI Coverage: {np.mean(cv_cc_coverages_aki):.1%} +/- "
      f"{np.std(cv_cc_coverages_aki)*100:.1f}%")

# ================================================================
# STEP 12: CLINICAL RISK STRATIFICATION
# ================================================================
print("\n" + "=" * 65)
print("STEP 12: Conformal risk stratification")
print("=" * 65)

# Use class-conditional sets for clinical stratification
risk_groups = []
for i, ps in enumerate(cc_prediction_sets):
    p = best_prob_test[i]
    if ps == {0}:
        risk_groups.append('Low Risk')
    elif ps == {1}:
        risk_groups.append('High Risk')
    elif len(ps) == 2:
        risk_groups.append('Uncertain')
    else:
        risk_groups.append('Uncertain')

risk_df = pd.DataFrame({
    'risk_group': risk_groups, 'aki': y_test, 'prob': best_prob_test
})

print("\nClass-conditional risk stratification:")
for g in ['Low Risk', 'Uncertain', 'High Risk']:
    mask = risk_df['risk_group'] == g
    if mask.sum() == 0: continue
    n_g = mask.sum()
    aki_r = risk_df.loc[mask, 'aki'].mean()
    mp = risk_df.loc[mask, 'prob'].mean()
    print(f"  {g:12s}: n={n_g:>6,} ({n_g/len(y_test):5.1%}), "
          f"AKI rate={aki_r:.1%}, mean P(AKI)={mp:.3f}")

# Decision curve
thresholds = np.arange(0.05, 0.95, 0.01)
nb_model = []
nb_all = []
for t in thresholds:
    pred_pos = best_prob_test >= t
    tp = (pred_pos & (y_test == 1)).sum()
    fp = (pred_pos & (y_test == 0)).sum()
    n = len(y_test)
    nb_model.append((tp/n) - (fp/n) * (t/(1-t)))
    prev = y_test.mean()
    nb_all.append(prev - (1-prev) * t/(1-t))

# ================================================================
# STEP 13: FIGURES
# ================================================================
print("\n" + "=" * 65)
print("STEP 13: Generating figures")
print("=" * 65)

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11,
                     'axes.titlesize': 12, 'figure.dpi': 300})

# --- Fig 1: ROC + PRC ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['prob_test'])
    ax.plot(fpr, tpr, label=f"{name} ({res['AUC']:.3f})", linewidth=1.5)
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(a) ROC Curves'); ax.legend(fontsize=7, loc='lower right')

ax = axes[1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['prob_test'])
    ax.plot(rec, prec, label=f"{name} ({res['AUPRC']:.3f})", linewidth=1.5)
ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.3,
           label=f'Prevalence={y_test.mean():.2f}')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('(b) Precision-Recall Curves'); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "fig1_roc_prc.png", bbox_inches='tight')
plt.close()
print("  fig1_roc_prc.png")

# --- Fig 2: Conformal sets ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Marginal
ax = axes[0]
cats = ['Singleton\n{0}', 'Singleton\n{1}', 'Both\n{0,1}', 'Empty']
cnts = [sum(1 for ps in prediction_sets if ps=={0}),
        sum(1 for ps in prediction_sets if ps=={1}),
        sum(1 for ps in prediction_sets if ps=={0,1}),
        sum(1 for ps in prediction_sets if len(ps)==0)]
colors = ['#2ecc71','#e74c3c','#f39c12','#95a5a6']
bars = ax.bar(cats, cnts, color=colors, edgecolor='black', linewidth=0.5)
for b,c in zip(bars,cnts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+10,
            f'{c:,}', ha='center', fontsize=8)
ax.set_ylabel('Count')
ax.set_title(f'(a) Marginal Prediction Sets (n={len(y_test):,})')

# Class-conditional
ax = axes[1]
cc_cats = ['Singleton\n{0}', 'Singleton\n{1}', 'Both\n{0,1}', 'Empty']
cc_cnts = [sum(1 for ps in cc_prediction_sets if ps=={0}),
           sum(1 for ps in cc_prediction_sets if ps=={1}),
           sum(1 for ps in cc_prediction_sets if ps=={0,1}),
           sum(1 for ps in cc_prediction_sets if len(ps)==0)]
bars = ax.bar(cc_cats, cc_cnts, color=colors, edgecolor='black', linewidth=0.5)
for b,c in zip(bars,cc_cnts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+10,
            f'{c:,}', ha='center', fontsize=8)
ax.set_ylabel('Count')
ax.set_title('(b) Class-Conditional Sets')

# Coverage comparison
ax = axes[2]
labels = ['Marginal', 'Class-Cond.']
cov_overall = [coverage*100, cc_coverage*100]
cov_aki_vals = [cov_aki*100, cc_cov_aki*100]
cov_noaki_vals = [cov_noaki*100, cc_cov_noaki*100]
x = np.arange(len(labels))
w = 0.25
ax.bar(x-w, cov_overall, w, label='Overall', color='#3498db', edgecolor='black', linewidth=0.5)
ax.bar(x, cov_aki_vals, w, label='AKI=1', color='#e74c3c', edgecolor='black', linewidth=0.5)
ax.bar(x+w, cov_noaki_vals, w, label='AKI=0', color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', linewidth=1.5, label='90% target')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Coverage (%)'); ax.set_title('(c) Coverage Comparison')
ax.legend(fontsize=7); ax.set_ylim(50, 105)
plt.tight_layout()
plt.savefig(FIG / "fig2_conformal.png", bbox_inches='tight')
plt.close()
print("  fig2_conformal.png")

# --- Fig 3: Feature importance ---
fig, ax = plt.subplots(figsize=(8, 6))
top_n = 20
top_feats = feat_imp.head(top_n).iloc[::-1]
colors_fi = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
ax.barh(range(top_n), top_feats['importance'].values,
        color=colors_fi, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats['feature'].values, fontsize=7)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} Features ({best_name})')
plt.tight_layout()
plt.savefig(FIG / "fig3_importance.png", bbox_inches='tight')
plt.close()
print("  fig3_importance.png")

# --- Fig 4: Decision curve ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, nb_model, 'b-', linewidth=2, label=best_name)
ax.plot(thresholds, nb_all, 'r--', linewidth=1, label='Treat All')
ax.axhline(y=0, color='k', linestyle=':', linewidth=1, label='Treat None')
ax.set_xlabel('Threshold Probability'); ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Analysis')
ax.legend(fontsize=9); ax.set_xlim(0.05, 0.80)
ax.set_ylim(-0.05, max(nb_model)*1.1 if max(nb_model) > 0 else 0.2)
plt.tight_layout()
plt.savefig(FIG / "fig4_decision_curve.png", bbox_inches='tight')
plt.close()
print("  fig4_decision_curve.png")

# --- Fig 5: Mondrian ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By gender
ax = axes[0]
g_groups = sorted([k for k in mondrian_results if k.startswith('gender')])
g_labels = ['Female' if '0' in k else 'Male' for k in g_groups]
g_covs = [mondrian_results[k]['coverage']*100 for k in g_groups]
bars = ax.bar(g_labels, g_covs, color=['#e74c3c','#3498db'],
              edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', linewidth=1.5, label='90% target')
for b,c in zip(bars,g_covs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f'{c:.1f}%', ha='center', fontsize=9)
ax.set_ylabel('Coverage (%)'); ax.set_title('(a) Coverage by Gender')
ax.legend(fontsize=8); ax.set_ylim(0, 110)

# By age
ax = axes[1]
a_groups = sorted([k for k in mondrian_results if k.startswith('age')])
a_labels = [k.split('=')[1] for k in a_groups]
a_covs = [mondrian_results[k]['coverage']*100 for k in a_groups]
colors_a = ['#2ecc71','#f39c12','#e74c3c'][:len(a_groups)]
bars = ax.bar(a_labels, a_covs, color=colors_a,
              edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', linewidth=1.5, label='90% target')
for b,c in zip(bars,a_covs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f'{c:.1f}%', ha='center', fontsize=9)
ax.set_ylabel('Coverage (%)'); ax.set_title('(b) Coverage by Age Group')
ax.legend(fontsize=8); ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(FIG / "fig5_mondrian.png", bbox_inches='tight')
plt.close()
print("  fig5_mondrian.png")

# --- Fig 6: CV stability ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.bar(range(1,6), cv_aucs, color='#3498db', edgecolor='black', linewidth=0.5)
ax.axhline(y=np.mean(cv_aucs), color='navy', linestyle='--',
           label=f'Mean={np.mean(cv_aucs):.3f}')
ax.set_xlabel('Fold'); ax.set_ylabel('AUC')
ax.set_title('(a) AUC Across Folds'); ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.0)

ax = axes[1]
ax.bar(range(1,6), [c*100 for c in cv_coverages],
       color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', label='90% target')
ax.axhline(y=np.mean(cv_coverages)*100, color='green', linestyle=':',
           label=f'Mean={np.mean(cv_coverages):.1%}')
ax.set_xlabel('Fold'); ax.set_ylabel('Coverage (%)')
ax.set_title('(b) Coverage Across Folds'); ax.legend(fontsize=8)
ax.set_ylim(70, 100)
plt.tight_layout()
plt.savefig(FIG / "fig6_cv_stability.png", bbox_inches='tight')
plt.close()
print("  fig6_cv_stability.png")

# --- Fig 8: Calibration comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (label, probs) in zip(axes, [('Raw', p_raw), ('Platt Scaling', p_platt),
                                       ('Isotonic Reg.', p_iso)]):
    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10,
                                                 strategy='uniform')
    ax.plot(mean_pred, fraction_pos, 'bo-', linewidth=1.5, markersize=6)
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    brier_c = brier_score_loss(y_test, probs)
    ax.set_title(f'{label}\n(Brier={brier_c:.4f})')
    ax.set_xlabel('Mean Predicted'); ax.set_ylabel('Fraction Positive')
plt.tight_layout()
plt.savefig(FIG / "fig8_calibration.png", bbox_inches='tight')
plt.close()
print("  fig8_calibration.png")

# ================================================================
# STEP 14: SAVE ALL RESULTS
# ================================================================
print("\n" + "=" * 65)
print("STEP 14: Saving results")
print("=" * 65)

model_table = pd.DataFrame({
    name: {'AUC': r['AUC'], 'AUPRC': r['AUPRC'],
           'Brier': r['Brier'], 'F1': r['F1'], 'Acc': r['Accuracy']}
    for name, r in results.items()
}).T
model_table.to_csv(RES / "model_comparison.csv")

summary = {
    'dataset': 'PhysioNet 2019 Sepsis Challenge',
    'n_patients': int(len(df)),
    'n_aki': int(df['aki_label'].sum()),
    'aki_prevalence': float(df['aki_label'].mean()),
    'n_features': len(feature_names),
    'n_train': int(len(X_train)),
    'n_cal': int(len(X_cal)),
    'n_test': int(len(X_test)),
    'best_model': best_name,
    'best_auc': float(results[best_name]['AUC']),
    'best_auprc': float(results[best_name]['AUPRC']),
    'alpha': alpha,
    'q_hat': float(q_hat),
    'marginal_coverage': float(coverage),
    'marginal_cov_aki': float(cov_aki),
    'marginal_cov_noaki': float(cov_noaki),
    'cc_coverage': float(cc_coverage),
    'cc_cov_aki': float(cc_cov_aki),
    'cc_cov_noaki': float(cc_cov_noaki),
    'cv_auc_mean': float(np.mean(cv_aucs)),
    'cv_auc_std': float(np.std(cv_aucs)),
    'cv_coverage_mean': float(np.mean(cv_coverages)),
    'cv_coverage_std': float(np.std(cv_coverages)),
}
with open(RES / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(mondrian_results).T.to_csv(RES / "mondrian_results.csv")

cohort_df = pd.DataFrame(cohort_stats).T
cohort_df.to_csv(RES / "cohort_stats.csv")

risk_summary = risk_df.groupby('risk_group').agg(
    n=('aki', 'count'), aki_rate=('aki', 'mean'),
    mean_prob=('prob', 'mean')).round(3)
risk_summary.to_csv(RES / "risk_stratification.csv")

print("  All results saved.")

print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)
print(f"  Dataset: {len(df):,} real ICU patients (PhysioNet 2019)")
print(f"  AKI prevalence: {df['aki_label'].mean():.1%}")
print(f"  Best model: {best_name} (AUC={results[best_name]['AUC']:.3f})")
print(f"  Marginal coverage: {coverage:.1%}")
print(f"  Class-conditional: AKI={cc_cov_aki:.1%}, NoAKI={cc_cov_noaki:.1%}")
print(f"  CV: AUC={np.mean(cv_aucs):.3f}+/-{np.std(cv_aucs):.3f}")
