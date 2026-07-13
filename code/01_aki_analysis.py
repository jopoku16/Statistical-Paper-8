"""
AKI Prediction with Conformal Prediction — Full Analysis Pipeline
Statistical Paper 8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
import json

ROOT = Path(r"C:\Users\19565\Desktop\Statistical Paper 8")
DATA = ROOT / "data"
FIG  = ROOT / "figures"
RES  = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. LOAD DATA & CONSTRUCT AKI LABEL
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load data and construct AKI label")
print("=" * 60)

df = pd.read_csv(DATA / "sepsis_icu_synthetic.csv")
print(f"Raw data: {df.shape[0]} patients, {df.shape[1]} columns")

# AKI definition (KDIGO Stage 2+ simplified for single-timepoint ICU data):
#   - Creatinine >= 2.5 mg/dL in patients WITHOUT pre-existing CKD
#   - Creatinine >= 3.5 mg/dL in patients WITH pre-existing CKD
#   (captures acute severe elevation; yields ~20% prevalence matching ICU literature)

has_ckd = df['chronic_kidney_disease'] == 1
cr = df['creatinine'].copy()

aki_label = pd.Series(0, index=df.index)
aki_label[(~has_ckd) & (cr >= 2.5)] = 1
aki_label[(has_ckd) & (cr >= 3.5)] = 1
aki_label[cr.isna()] = np.nan

df['aki_label'] = aki_label

# Drop patients with missing creatinine (can't define AKI)
n_before = len(df)
df = df.dropna(subset=['aki_label']).copy()
df['aki_label'] = df['aki_label'].astype(int)
print(f"After removing missing creatinine: {len(df)} patients ({n_before - len(df)} dropped)")
print(f"AKI prevalence: {df['aki_label'].mean():.1%} ({df['aki_label'].sum()}/{len(df)})")

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature engineering")
print("=" * 60)

# Remove identifiers, outcome-related, and leaking variables
drop_cols = [
    'subject_id', 'aki_label',
    'creatinine',  # direct outcome leakage
    'sepsis_label',  # different outcome
    'readmission_30day',  # future info
    'icu_los_hours',  # future info
]

# Encode categoricals
cat_cols = ['gender', 'ethnicity', 'insurance', 'hospital_admit_source']
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
drop_cols += cat_cols

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Features: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df['aki_label'].values

# Impute missing values (median for continuous)
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

# Derived features
X_imp['shock_index'] = X_imp['hr_mean'] / X_imp['sbp_mean'].clip(lower=60)
X_imp['hr_sbp_product'] = X_imp['hr_mean'] * X_imp['sbp_mean']
X_imp['lactate_sofa'] = X_imp['lactate_mmol'] * X_imp['sofa_score']
X_imp['n_comorbidities'] = X_imp[['diabetes', 'hypertension', 'chf', 'copd',
    'chronic_kidney_disease', 'liver_disease', 'immunosuppression',
    'cad', 'atrial_fibrillation', 'cancer_active']].sum(axis=1)

feature_cols_final = list(X_imp.columns)
print(f"Features after engineering: {len(feature_cols_final)}")

# ──────────────────────────────────────────────
# 3. TRAIN/CALIBRATION/TEST SPLIT
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Data splitting (60/20/20)")
print("=" * 60)

# 60% train, 20% calibration, 20% test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_imp, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"Train: {len(X_train)} (AKI={y_train.sum()}, {y_train.mean():.1%})")
print(f"Calibration: {len(X_cal)} (AKI={y_cal.sum()}, {y_cal.mean():.1%})")
print(f"Test: {len(X_test)} (AKI={y_test.sum()}, {y_test.mean():.1%})")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_cal_sc = scaler.transform(X_cal)
X_test_sc = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 4. TRAIN MULTIPLE ML MODELS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Train ML models")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, C=0.1, penalty='l2', solver='lbfgs', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    ),
}

# Try XGBoost if available
try:
    from xgboost import XGBClassifier
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
except ImportError:
    print("XGBoost not available, skipping")

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'Logistic Regression':
        model.fit(X_train_sc, y_train)
        prob_cal = model.predict_proba(X_cal_sc)[:, 1]
        prob_test = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        prob_cal = model.predict_proba(X_cal.values)[:, 1]
        prob_test = model.predict_proba(X_test.values)[:, 1]

    auc = roc_auc_score(y_test, prob_test)
    ap = average_precision_score(y_test, prob_test)
    brier = brier_score_loss(y_test, prob_test)
    pred_test = (prob_test >= 0.5).astype(int)
    f1 = f1_score(y_test, pred_test)
    acc = accuracy_score(y_test, pred_test)

    results[name] = {
        'AUC': auc, 'AUPRC': ap, 'Brier': brier,
        'F1': f1, 'Accuracy': acc,
        'prob_cal': prob_cal, 'prob_test': prob_test
    }
    trained_models[name] = model
    print(f"  AUC={auc:.3f}, AUPRC={ap:.3f}, Brier={brier:.3f}, F1={f1:.3f}")

# Identify best model by AUC
best_name = max(results, key=lambda k: results[k]['AUC'])
print(f"\nBest model: {best_name} (AUC={results[best_name]['AUC']:.3f})")

# ──────────────────────────────────────────────
# 5. SPLIT CONFORMAL PREDICTION
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Split conformal prediction")
print("=" * 60)

alpha = 0.10  # 90% coverage target
m = len(y_cal)

# Nonconformity scores for classification: 1 - p(true class)
best_prob_cal = results[best_name]['prob_cal']
best_prob_test = results[best_name]['prob_test']

# Adaptive Prediction Sets (APS) conformal
scores_cal = np.where(y_cal == 1, 1 - best_prob_cal, best_prob_cal)

# Quantile for coverage guarantee
q_level = np.ceil((1 - alpha) * (m + 1)) / m
q_hat = np.quantile(scores_cal, q_level, method='higher')

print(f"Calibration set size: m = {m}")
print(f"Target coverage: {1 - alpha:.0%}")
print(f"Adjusted quantile level: {q_level:.4f}")
print(f"Conformal threshold q_hat = {q_hat:.4f}")
print(f"Finite-sample coverage bound: [{1-alpha:.2f}, {1-alpha + 1/(m+1):.4f}]")

# Generate prediction sets for test data
prediction_sets = []
for p in best_prob_test:
    pset = set()
    if (1 - p) <= q_hat:
        pset.add(1)  # AKI included
    if p <= q_hat:
        pset.add(0)  # No AKI included
    prediction_sets.append(pset)

# Coverage and set sizes
covered = sum(1 for i, ps in enumerate(prediction_sets) if y_test[i] in ps)
coverage = covered / len(y_test)

set_sizes = [len(ps) for ps in prediction_sets]
empty_sets = sum(1 for s in set_sizes if s == 0)
singleton_sets = sum(1 for s in set_sizes if s == 1)
both_sets = sum(1 for s in set_sizes if s == 2)

print(f"\nTest set results:")
print(f"  Coverage: {coverage:.1%} (target: {1-alpha:.0%})")
print(f"  Empty sets: {empty_sets} ({empty_sets/len(y_test):.1%})")
print(f"  Singletons (decisive): {singleton_sets} ({singleton_sets/len(y_test):.1%})")
print(f"  Both-class sets (uncertain): {both_sets} ({both_sets/len(y_test):.1%})")

# Coverage by subgroup
aki_mask = y_test == 1
cov_aki = sum(1 for i in range(len(y_test)) if aki_mask[i] and y_test[i] in prediction_sets[i]) / max(aki_mask.sum(), 1)
cov_noaki = sum(1 for i in range(len(y_test)) if not aki_mask[i] and y_test[i] in prediction_sets[i]) / max((~aki_mask).sum(), 1)

print(f"  Coverage (AKI=1): {cov_aki:.1%}")
print(f"  Coverage (AKI=0): {cov_noaki:.1%}")

# ──────────────────────────────────────────────
# 6. MONDRIAN CONFORMAL (GROUP-CONDITIONAL)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Mondrian conformal prediction (group-conditional)")
print("=" * 60)

# Group by CKD status and age group
cal_df = pd.DataFrame({
    'prob': best_prob_cal,
    'y': y_cal,
    'score': scores_cal,
    'ckd': X_cal['chronic_kidney_disease'].values,
    'age': X_cal['age_enc' if 'age_enc' in X_cal.columns else 'age'].values
})
cal_df['age_group'] = pd.cut(cal_df['age'], bins=[0, 50, 65, 120], labels=['<50', '50-65', '>65'])

test_df = pd.DataFrame({
    'prob': best_prob_test,
    'y': y_test,
    'ckd': X_test['chronic_kidney_disease'].values,
    'age': X_test['age_enc' if 'age_enc' in X_test.columns else 'age'].values
})
test_df['age_group'] = pd.cut(test_df['age'], bins=[0, 50, 65, 120], labels=['<50', '50-65', '>65'])

mondrian_results = {}
for group_col in ['ckd', 'age_group']:
    print(f"\n  Mondrian by {group_col}:")
    groups = cal_df[group_col].unique()
    for g in sorted(groups, key=str):
        cal_mask = cal_df[group_col] == g
        test_mask = test_df[group_col] == g
        if cal_mask.sum() < 10 or test_mask.sum() < 5:
            continue

        g_scores = cal_df.loc[cal_mask, 'score'].values
        m_g = len(g_scores)
        q_g = np.quantile(g_scores,
                          np.ceil((1 - alpha) * (m_g + 1)) / m_g,
                          method='higher')

        g_probs = test_df.loc[test_mask, 'prob'].values
        g_y = test_df.loc[test_mask, 'y'].values

        g_psets = []
        for p in g_probs:
            ps = set()
            if (1 - p) <= q_g:
                ps.add(1)
            if p <= q_g:
                ps.add(0)
            g_psets.append(ps)

        g_cov = sum(1 for i, ps in enumerate(g_psets) if g_y[i] in ps) / len(g_y)
        g_sizes = np.mean([len(ps) for ps in g_psets])
        g_singleton = sum(1 for ps in g_psets if len(ps) == 1) / len(g_psets)

        key = f"{group_col}={g}"
        mondrian_results[key] = {
            'n_cal': m_g, 'n_test': len(g_y),
            'q_hat': q_g, 'coverage': g_cov,
            'avg_set_size': g_sizes, 'singleton_frac': g_singleton
        }
        print(f"    {key}: n_cal={m_g}, coverage={g_cov:.1%}, "
              f"avg_size={g_sizes:.2f}, q_hat={q_g:.4f}")

# ──────────────────────────────────────────────
# 7. CROSS-VALIDATION STABILITY
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: 5-fold cross-validation stability")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
cv_coverages = []

X_full = X_imp.values
y_full = y

for fold, (train_idx, test_idx) in enumerate(cv.split(X_full, y_full)):
    X_tr, X_te = X_full[train_idx], X_full[test_idx]
    y_tr, y_te = y_full[train_idx], y_full[test_idx]

    # Split train into train+cal
    X_tr2, X_ca2, y_tr2, y_ca2 = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=fold, stratify=y_tr
    )

    if best_name == 'Logistic Regression':
        sc2 = StandardScaler()
        X_tr2 = sc2.fit_transform(X_tr2)
        X_ca2 = sc2.transform(X_ca2)
        X_te = sc2.transform(X_te)

    mdl = trained_models[best_name].__class__(**trained_models[best_name].get_params())
    mdl.fit(X_tr2, y_tr2)

    p_ca = mdl.predict_proba(X_ca2)[:, 1]
    p_te = mdl.predict_proba(X_te)[:, 1]

    auc_f = roc_auc_score(y_te, p_te)
    cv_aucs.append(auc_f)

    # Conformal
    sc_ca = np.where(y_ca2 == 1, 1 - p_ca, p_ca)
    m_f = len(sc_ca)
    q_f = np.quantile(sc_ca, np.ceil((1 - alpha) * (m_f + 1)) / m_f, method='higher')

    cov_f = 0
    for i, p in enumerate(p_te):
        ps = set()
        if (1 - p) <= q_f:
            ps.add(1)
        if p <= q_f:
            ps.add(0)
        if y_te[i] in ps:
            cov_f += 1
    cov_f /= len(y_te)
    cv_coverages.append(cov_f)

    print(f"  Fold {fold+1}: AUC={auc_f:.3f}, Coverage={cov_f:.1%}")

print(f"\nCV Summary: AUC={np.mean(cv_aucs):.3f}±{np.std(cv_aucs):.3f}, "
      f"Coverage={np.mean(cv_coverages):.1%}±{np.std(cv_coverages)*100:.1f}%")

# ──────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature importance")
print("=" * 60)

if best_name == 'Gradient Boosting':
    importances = trained_models[best_name].feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols_final,
        'importance': importances
    }).sort_values('importance', ascending=False)
elif best_name == 'XGBoost':
    importances = trained_models[best_name].feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols_final,
        'importance': importances
    }).sort_values('importance', ascending=False)
elif best_name == 'Random Forest':
    importances = trained_models[best_name].feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols_final,
        'importance': importances
    }).sort_values('importance', ascending=False)
else:
    coefs = np.abs(trained_models[best_name].coef_[0])
    feat_imp = pd.DataFrame({
        'feature': feature_cols_final,
        'importance': coefs
    }).sort_values('importance', ascending=False)

print("Top 15 features:")
for _, row in feat_imp.head(15).iterrows():
    print(f"  {row['feature']:35s} {row['importance']:.4f}")

feat_imp.to_csv(RES / "feature_importance.csv", index=False)

# ──────────────────────────────────────────────
# 9. CLINICAL DECISION ANALYSIS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Clinical decision analysis")
print("=" * 60)

# Net benefit at different thresholds (decision curve analysis)
thresholds = np.arange(0.05, 0.95, 0.01)
nb_model = []
nb_all = []
for t in thresholds:
    pred_pos = best_prob_test >= t
    tp = ((pred_pos) & (y_test == 1)).sum()
    fp = ((pred_pos) & (y_test == 0)).sum()
    n = len(y_test)
    nb = (tp / n) - (fp / n) * (t / (1 - t))
    nb_model.append(nb)
    # Treat-all
    prevalence = y_test.mean()
    nb_all.append(prevalence - (1 - prevalence) * t / (1 - t))

# Risk stratification using conformal sets
risk_groups = []
for i, ps in enumerate(prediction_sets):
    p = best_prob_test[i]
    if len(ps) == 1 and 0 in ps:
        risk_groups.append('Low Risk')
    elif len(ps) == 2:
        risk_groups.append('Uncertain')
    elif len(ps) == 1 and 1 in ps:
        risk_groups.append('High Risk')
    else:
        risk_groups.append('Empty')

risk_df = pd.DataFrame({
    'risk_group': risk_groups,
    'aki': y_test,
    'prob': best_prob_test
})

print("\nConformal risk stratification:")
for g in ['Low Risk', 'Uncertain', 'High Risk', 'Empty']:
    mask = risk_df['risk_group'] == g
    if mask.sum() == 0:
        continue
    n_g = mask.sum()
    aki_rate = risk_df.loc[mask, 'aki'].mean()
    mean_prob = risk_df.loc[mask, 'prob'].mean()
    print(f"  {g:12s}: n={n_g:4d} ({n_g/len(y_test):5.1%}), "
          f"AKI rate={aki_rate:.1%}, mean P(AKI)={mean_prob:.3f}")

# ──────────────────────────────────────────────
# 10. FIGURES
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Generating figures")
print("=" * 60)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'figure.dpi': 300
})

# --- Figure 1: ROC curves for all models ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['prob_test'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.3f})", linewidth=1.5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('(a) ROC Curves')
ax.legend(fontsize=8, loc='lower right')

ax = axes[1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['prob_test'])
    ax.plot(rec, prec, label=f"{name} (AUPRC={res['AUPRC']:.3f})", linewidth=1.5)
ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.3, label=f'Prevalence={y_test.mean():.2f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('(b) Precision-Recall Curves')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(FIG / "fig1_roc_prc.png", bbox_inches='tight')
plt.close()
print("  fig1_roc_prc.png saved")

# --- Figure 2: Conformal prediction sets visualization ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel a: Prediction set composition
ax = axes[0]
categories = ['Singleton\n{0}', 'Singleton\n{1}', 'Both\n{0,1}', 'Empty\nset']
counts = [
    sum(1 for ps in prediction_sets if ps == {0}),
    sum(1 for ps in prediction_sets if ps == {1}),
    sum(1 for ps in prediction_sets if ps == {0, 1}),
    sum(1 for ps in prediction_sets if len(ps) == 0)
]
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(count), ha='center', fontsize=9)
ax.set_ylabel('Count')
ax.set_title(f'(a) Prediction Set Composition (n={len(y_test)})')

# Panel b: Coverage by risk group
ax = axes[1]
groups_ordered = ['Low Risk', 'Uncertain', 'High Risk']
cov_by_group = []
n_by_group = []
for g in groups_ordered:
    mask = np.array(risk_groups) == g
    if mask.sum() > 0:
        g_covered = sum(1 for i in range(len(y_test))
                       if mask[i] and y_test[i] in prediction_sets[i])
        cov_by_group.append(g_covered / mask.sum())
        n_by_group.append(mask.sum())
    else:
        cov_by_group.append(0)
        n_by_group.append(0)

bar_colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(groups_ordered, [c * 100 for c in cov_by_group],
              color=bar_colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', linewidth=1.5, label='90% target')
for bar, cov, n in zip(bars, cov_by_group, n_by_group):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{cov:.1%}\n(n={n})', ha='center', fontsize=8)
ax.set_ylabel('Coverage (%)')
ax.set_title('(b) Coverage by Risk Group')
ax.legend(fontsize=8)
ax.set_ylim(0, 110)

# Panel c: Predicted probability vs actual AKI rate
ax = axes[2]
n_bins = 10
prob_bins = np.linspace(0, 1, n_bins + 1)
bin_centers = []
bin_aki_rates = []
bin_sizes = []
for i in range(n_bins):
    mask = (best_prob_test >= prob_bins[i]) & (best_prob_test < prob_bins[i+1])
    if mask.sum() > 5:
        bin_centers.append((prob_bins[i] + prob_bins[i+1]) / 2)
        bin_aki_rates.append(y_test[mask].mean())
        bin_sizes.append(mask.sum())

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
ax.scatter(bin_centers, bin_aki_rates, s=[b*2 for b in bin_sizes],
          c='#3498db', alpha=0.7, edgecolors='black', linewidth=0.5)
ax.plot(bin_centers, bin_aki_rates, 'b-', alpha=0.5)
ax.set_xlabel('Predicted P(AKI)')
ax.set_ylabel('Observed AKI Rate')
ax.set_title('(c) Calibration Plot')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG / "fig2_conformal.png", bbox_inches='tight')
plt.close()
print("  fig2_conformal.png saved")

# --- Figure 3: Feature importance ---
fig, ax = plt.subplots(figsize=(8, 6))
top_n = 20
top_feats = feat_imp.head(top_n).iloc[::-1]
colors_fi = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
ax.barh(range(top_n), top_feats['importance'].values,
        color=colors_fi, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats['feature'].values, fontsize=8)
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top {top_n} Features ({best_name})')
plt.tight_layout()
plt.savefig(FIG / "fig3_importance.png", bbox_inches='tight')
plt.close()
print("  fig3_importance.png saved")

# --- Figure 4: Decision curve analysis ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, nb_model, 'b-', linewidth=2, label=best_name)
ax.plot(thresholds, nb_all, 'r--', linewidth=1, label='Treat All')
ax.axhline(y=0, color='k', linestyle=':', linewidth=1, label='Treat None')
ax.set_xlabel('Threshold Probability')
ax.set_ylabel('Net Benefit')
ax.set_title('Decision Curve Analysis')
ax.legend(fontsize=9)
ax.set_xlim(0.05, 0.80)
ax.set_ylim(-0.05, max(nb_model) * 1.1)
plt.tight_layout()
plt.savefig(FIG / "fig4_decision_curve.png", bbox_inches='tight')
plt.close()
print("  fig4_decision_curve.png saved")

# --- Figure 5: Mondrian conformal results ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By CKD status
ax = axes[0]
ckd_groups = [k for k in mondrian_results if k.startswith('ckd')]
ckd_labels = [k.split('=')[1] for k in ckd_groups]
ckd_labels = ['No CKD' if l == '0' else 'CKD' for l in ckd_labels]
ckd_covs = [mondrian_results[k]['coverage'] * 100 for k in ckd_groups]
ckd_sizes = [mondrian_results[k]['avg_set_size'] for k in ckd_groups]

x_pos = np.arange(len(ckd_groups))
bars = ax.bar(x_pos, ckd_covs, color=['#3498db', '#e74c3c'],
              edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', linewidth=1.5, label='90% target')
for bar, cov, sz in zip(bars, ckd_covs, ckd_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{cov:.1f}%\nsize={sz:.2f}', ha='center', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(ckd_labels)
ax.set_ylabel('Coverage (%)')
ax.set_title('(a) Mondrian Coverage by CKD Status')
ax.legend(fontsize=8)
ax.set_ylim(0, 110)

# By age group
ax = axes[1]
age_groups = [k for k in mondrian_results if k.startswith('age')]
age_labels = [k.split('=')[1] for k in age_groups]
age_covs = [mondrian_results[k]['coverage'] * 100 for k in age_groups]
age_sizes = [mondrian_results[k]['avg_set_size'] for k in age_groups]

x_pos = np.arange(len(age_groups))
colors_age = ['#2ecc71', '#f39c12', '#e74c3c'][:len(age_groups)]
bars = ax.bar(x_pos, age_covs, color=colors_age,
              edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', linewidth=1.5, label='90% target')
for bar, cov, sz in zip(bars, age_covs, age_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{cov:.1f}%\nsize={sz:.2f}', ha='center', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(age_labels)
ax.set_ylabel('Coverage (%)')
ax.set_title('(b) Mondrian Coverage by Age Group')
ax.legend(fontsize=8)
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig(FIG / "fig5_mondrian.png", bbox_inches='tight')
plt.close()
print("  fig5_mondrian.png saved")

# --- Figure 6: CV stability ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.bar(range(1, 6), cv_aucs, color='#3498db', edgecolor='black', linewidth=0.5)
ax.axhline(y=np.mean(cv_aucs), color='navy', linestyle='--',
           label=f'Mean={np.mean(cv_aucs):.3f}')
ax.set_xlabel('Fold')
ax.set_ylabel('AUC')
ax.set_title('(a) AUC Across Folds')
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.0)

ax = axes[1]
ax.bar(range(1, 6), [c * 100 for c in cv_coverages],
       color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.axhline(y=90, color='navy', linestyle='--', label='90% target')
ax.axhline(y=np.mean(cv_coverages)*100, color='green', linestyle=':',
           label=f'Mean={np.mean(cv_coverages):.1%}')
ax.set_xlabel('Fold')
ax.set_ylabel('Coverage (%)')
ax.set_title('(b) Conformal Coverage Across Folds')
ax.legend(fontsize=8)
ax.set_ylim(70, 100)

plt.tight_layout()
plt.savefig(FIG / "fig6_cv_stability.png", bbox_inches='tight')
plt.close()
print("  fig6_cv_stability.png saved")

# ──────────────────────────────────────────────
# 11. SAVE ALL RESULTS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Saving results")
print("=" * 60)

# Model comparison table
model_table = pd.DataFrame({
    name: {
        'AUC': r['AUC'], 'AUPRC': r['AUPRC'],
        'Brier Score': r['Brier'], 'F1': r['F1'],
        'Accuracy': r['Accuracy']
    } for name, r in results.items()
}).T
model_table.to_csv(RES / "model_comparison.csv")
print("  model_comparison.csv saved")

# Conformal results
conformal_summary = {
    'alpha': alpha,
    'calibration_size': m,
    'q_hat': float(q_hat),
    'coverage': float(coverage),
    'coverage_aki': float(cov_aki),
    'coverage_noaki': float(cov_noaki),
    'singleton_fraction': singleton_sets / len(y_test),
    'uncertain_fraction': both_sets / len(y_test),
    'cv_auc_mean': float(np.mean(cv_aucs)),
    'cv_auc_std': float(np.std(cv_aucs)),
    'cv_coverage_mean': float(np.mean(cv_coverages)),
    'cv_coverage_std': float(np.std(cv_coverages)),
    'best_model': best_name,
    'n_patients': len(df),
    'n_train': len(X_train),
    'n_cal': len(X_cal),
    'n_test': len(X_test),
    'aki_prevalence': float(df['aki_label'].mean()),
    'n_features': len(feature_cols_final),
}
with open(RES / "conformal_summary.json", 'w') as f:
    json.dump(conformal_summary, f, indent=2)
print("  conformal_summary.json saved")

# Mondrian results
mondrian_df = pd.DataFrame(mondrian_results).T
mondrian_df.to_csv(RES / "mondrian_results.csv")
print("  mondrian_results.csv saved")

# Risk stratification
risk_summary = risk_df.groupby('risk_group').agg(
    n=('aki', 'count'),
    aki_rate=('aki', 'mean'),
    mean_prob=('prob', 'mean')
).round(3)
risk_summary.to_csv(RES / "risk_stratification.csv")
print("  risk_stratification.csv saved")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nKey findings:")
print(f"  Dataset: {len(df)} ICU patients, AKI prevalence {df['aki_label'].mean():.1%}")
print(f"  Best model: {best_name} (AUC={results[best_name]['AUC']:.3f})")
print(f"  Conformal coverage: {coverage:.1%} (target: 90%)")
print(f"  Decisive predictions: {singleton_sets/len(y_test):.1%}")
print(f"  CV stability: AUC={np.mean(cv_aucs):.3f}±{np.std(cv_aucs):.3f}")
