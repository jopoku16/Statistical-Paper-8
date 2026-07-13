"""
Add deep learning (MLP) to model benchmark and bootstrap CIs.
Reuses same data pipeline as 03_enhanced_analysis.py.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, accuracy_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time

OUT = r"C:\Users\19565\Desktop\Statistical Paper 8"
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")

# ================================================================
# STEP 1: Load and preprocess (same as 03)
# ================================================================
print("=" * 65)
print("STEP 1: Load data and preprocess")
print("=" * 65)

t0 = time.time()
df = pd.read_csv(os.path.join(OUT, "data", "physionet", "Dataset.csv"))
print(f"Loaded {len(df):,} records in {time.time()-t0:.1f}s")

bidmc_ids = set(df.loc[df['Patient_ID'] <= 50000, 'Patient_ID'].unique())
emory_ids = set(df.loc[df['Patient_ID'] > 50000, 'Patient_ID'].unique())

cr_data = df.dropna(subset=['Creatinine']).groupby('Patient_ID')['Creatinine']
cr_first = cr_data.first()
cr_max = cr_data.max()
cr_count = cr_data.count()

aki_labels = pd.Series(0, index=cr_first.index)
fold_change = cr_max / cr_first
abs_change = cr_max - cr_first
aki_labels[(fold_change >= 1.5) | (abs_change >= 0.3)] = 1

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

leak_cols = [c for c in feat.columns if 'Creatinine' in c]
feat.drop(columns=leak_cols, inplace=True, errors='ignore')

common = feat.index.intersection(aki_labels.index)
feat = feat.loc[common]
y_all = aki_labels.loc[common]

bidmc_mask = feat.index.isin(bidmc_ids)
emory_mask = feat.index.isin(emory_ids)

X_bidmc = feat.loc[bidmc_mask]
y_bidmc = y_all.loc[bidmc_mask]
X_emory = feat.loc[emory_mask]
y_emory = y_all.loc[emory_mask]

X_train, X_cal, y_train, y_cal = train_test_split(
    X_bidmc, y_bidmc, test_size=0.30, stratify=y_bidmc, random_state=42
)

num_cols = X_train.select_dtypes(include=[np.number]).columns
X_train = X_train[num_cols].dropna(axis=1, how='all')
kept_cols = X_train.columns
X_cal = X_cal[kept_cols]
X_emory_num = X_emory[kept_cols]

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=kept_cols, index=X_train.index)
X_cal_imp = pd.DataFrame(imputer.transform(X_cal),
                         columns=kept_cols, index=X_cal.index)
X_emory_imp = pd.DataFrame(imputer.transform(X_emory_num),
                           columns=kept_cols, index=X_emory.index)

X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp),
                          columns=kept_cols, index=X_train.index)
X_cal_sc = pd.DataFrame(scaler.transform(X_cal_imp),
                        columns=kept_cols, index=X_cal.index)
X_emory_sc = pd.DataFrame(scaler.transform(X_emory_imp),
                          columns=kept_cols, index=X_emory.index)

print(f"Train: {len(X_train):,}, Cal: {len(X_cal):,}, External: {len(X_emory):,}")
print(f"Features: {len(kept_cols)}")

# ================================================================
# STEP 2: Train all 6 models (5 original + MLP)
# ================================================================
print("\n" + "=" * 65)
print("STEP 2: Train 6 ML models")
print("=" * 65)

models = {}

t1 = time.time()
models['Logistic Regression'] = LogisticRegression(
    C=0.1, max_iter=2000, solver='saga', random_state=42
)
models['Logistic Regression'].fit(X_train_sc, y_train)
print(f"  LR trained ({time.time()-t1:.1f}s)")

t1 = time.time()
models['Random Forest'] = RandomForestClassifier(
    n_estimators=500, max_depth=10, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
models['Random Forest'].fit(X_train_imp, y_train)
print(f"  RF trained ({time.time()-t1:.1f}s)")

t1 = time.time()
models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
models['Gradient Boosting'].fit(X_train_imp, y_train)
print(f"  GB trained ({time.time()-t1:.1f}s)")

t1 = time.time()
models['XGBoost'] = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, eval_metric='logloss',
    use_label_encoder=False
)
models['XGBoost'].fit(X_train_imp, y_train)
print(f"  XGBoost trained ({time.time()-t1:.1f}s)")

t1 = time.time()
models['LightGBM'] = lgb.LGBMClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, verbose=-1
)
models['LightGBM'].fit(X_train_imp, y_train)
print(f"  LightGBM trained ({time.time()-t1:.1f}s)")

t1 = time.time()
models['MLP'] = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42
)
models['MLP'].fit(X_train_sc, y_train)
print(f"  MLP trained ({time.time()-t1:.1f}s)")

# ================================================================
# STEP 3: Evaluate all on external data
# ================================================================
print("\n" + "=" * 65)
print("STEP 3: External validation (Emory)")
print("=" * 65)

results = {}
for name, model in models.items():
    X_test = X_emory_sc if name in ['Logistic Regression', 'MLP'] else X_emory_imp
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_emory, probs)
    auprc = average_precision_score(y_emory, probs)
    brier = brier_score_loss(y_emory, probs)
    f1 = f1_score(y_emory, preds)
    acc = accuracy_score(y_emory, preds)

    results[name] = {
        'AUC': auc, 'AUPRC': auprc, 'Brier': brier,
        'F1': f1, 'Accuracy': acc, 'probs': probs
    }
    print(f"  {name:25s} AUC={auc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.3f}  F1={f1:.3f}")

# ================================================================
# STEP 4: Bootstrap confidence intervals
# ================================================================
print("\n" + "=" * 65)
print("STEP 4: Bootstrap 95% CIs for AUC (1000 resamples)")
print("=" * 65)

n_boot = 1000
rng = np.random.RandomState(42)

bootstrap_results = {}
y_ext = np.array(y_emory)

for name in models:
    probs = results[name]['probs']
    boot_aucs = []
    for b in range(n_boot):
        idx = rng.choice(len(y_ext), size=len(y_ext), replace=True)
        if len(np.unique(y_ext[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_ext[idx], probs[idx]))

    boot_aucs = np.array(boot_aucs)
    ci_lo = np.percentile(boot_aucs, 2.5)
    ci_hi = np.percentile(boot_aucs, 97.5)
    bootstrap_results[name] = {
        'AUC': results[name]['AUC'],
        'CI_lo': ci_lo,
        'CI_hi': ci_hi
    }
    print(f"  {name:25s} AUC={results[name]['AUC']:.3f} (95% CI: {ci_lo:.3f}--{ci_hi:.3f})")

# Bootstrap CI for best model's conformal coverage
print("\nBootstrap CIs for XGBoost conformal coverage on external data...")

best_name = max(results, key=lambda k: results[k]['AUC'])
best_probs = results[best_name]['probs']

# Calibration scores from BIDMC cal set
best_model = models[best_name]
X_cal_use = X_cal_sc if best_name in ['Logistic Regression', 'MLP'] else X_cal_imp
cal_probs = best_model.predict_proba(X_cal_use)[:, 1]
y_cal_arr = np.array(y_cal)

# Marginal conformal
cal_scores = np.where(y_cal_arr == 1, 1 - cal_probs, cal_probs)
alpha = 0.10
m = len(cal_scores)
q_hat = np.quantile(cal_scores, np.ceil((1 - alpha) * (m + 1)) / m, method='higher')

# CC conformal
cal_scores_0 = 1 - cal_probs[y_cal_arr == 0]  # score for class 0: prob of wrong class
cal_scores_1 = 1 - cal_probs[y_cal_arr == 1]   # score for class 1: 1 - prob(1)
m0 = len(cal_scores_0)
m1 = len(cal_scores_1)
q0 = np.quantile(cal_scores_0, np.ceil((1 - alpha) * (m0 + 1)) / m0, method='higher')
q1 = np.quantile(cal_scores_1, np.ceil((1 - alpha) * (m1 + 1)) / m1, method='higher')

# External coverage
ext_probs = best_probs
y_ext_arr = np.array(y_emory)

# Marginal coverage
marg_in_set = []
for i in range(len(y_ext_arr)):
    p1 = ext_probs[i]
    p0 = 1 - p1
    s0 = 1 - p0  # = p1
    s1 = 1 - p1  # = p0
    pred_set = set()
    if s0 <= q_hat:
        pred_set.add(0)
    if s1 <= q_hat:
        pred_set.add(1)
    marg_in_set.append(y_ext_arr[i] in pred_set)
marg_in_set = np.array(marg_in_set)

# CC coverage
cc_in_set = []
for i in range(len(y_ext_arr)):
    p1 = ext_probs[i]
    p0 = 1 - p1
    pred_set = set()
    if p1 <= q0:  # score for class 0 = p1 <= q0
        pred_set.add(0)
    if (1 - p1) <= q1:  # score for class 1 = 1-p1 <= q1
        pred_set.add(1)
    cc_in_set.append(y_ext_arr[i] in pred_set)
cc_in_set = np.array(cc_in_set)

boot_marg_cov = []
boot_cc_aki_cov = []
for b in range(n_boot):
    idx = rng.choice(len(y_ext_arr), size=len(y_ext_arr), replace=True)
    boot_marg_cov.append(marg_in_set[idx].mean())
    aki_idx = idx[y_ext_arr[idx] == 1]
    if len(aki_idx) > 0:
        boot_cc_aki_cov.append(cc_in_set[aki_idx].mean())

boot_marg_cov = np.array(boot_marg_cov)
boot_cc_aki_cov = np.array(boot_cc_aki_cov)

marg_ci = (np.percentile(boot_marg_cov, 2.5), np.percentile(boot_marg_cov, 97.5))
cc_aki_ci = (np.percentile(boot_cc_aki_cov, 2.5), np.percentile(boot_cc_aki_cov, 97.5))

print(f"  Marginal coverage: {marg_in_set.mean():.1%} (95% CI: {marg_ci[0]:.1%}--{marg_ci[1]:.1%})")
print(f"  CC-AKI coverage:   {cc_in_set[y_ext_arr==1].mean():.1%} (95% CI: {cc_aki_ci[0]:.1%}--{cc_aki_ci[1]:.1%})")

# ================================================================
# STEP 5: Updated ROC/PRC figure with 6 models
# ================================================================
print("\n" + "=" * 65)
print("STEP 5: Generate updated figures")
print("=" * 65)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
model_order = ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
               'XGBoost', 'LightGBM', 'MLP']

for i, name in enumerate(model_order):
    probs = results[name]['probs']
    fpr, tpr, _ = roc_curve(y_emory, probs)
    auc = results[name]['AUC']
    ci = bootstrap_results[name]
    label = f"{name} ({auc:.3f} [{ci['CI_lo']:.3f}-{ci['CI_hi']:.3f}])"
    ax1.plot(fpr, tpr, color=colors[i], lw=2, label=label)

ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('(a) ROC Curves — External Validation', fontsize=13)
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

for i, name in enumerate(model_order):
    probs = results[name]['probs']
    prec, rec, _ = precision_recall_curve(y_emory, probs)
    auprc = results[name]['AUPRC']
    ax2.plot(rec, prec, color=colors[i], lw=2,
             label=f"{name} ({auprc:.3f})")

ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('(b) Precision-Recall Curves', fontsize=13)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=y_emory.mean(), color='gray', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig1_roc_prc.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Updated fig1_roc_prc.png (6 models with CIs)")

# ================================================================
# STEP 6: Save updated results
# ================================================================
print("\n" + "=" * 65)
print("STEP 6: Save results")
print("=" * 65)

# Update model comparison CSV
rows = []
for name in model_order:
    r = results[name]
    ci = bootstrap_results[name]
    rows.append({
        'Model': name,
        'AUC': round(r['AUC'], 3),
        'AUC_CI_lo': round(ci['CI_lo'], 3),
        'AUC_CI_hi': round(ci['CI_hi'], 3),
        'AUPRC': round(r['AUPRC'], 3),
        'Brier': round(r['Brier'], 3),
        'F1': round(r['F1'], 3),
        'Accuracy': round(r['Accuracy'], 3)
    })
pd.DataFrame(rows).to_csv(os.path.join(RES, 'model_comparison.csv'), index=False)
print("  model_comparison.csv updated")

# Save bootstrap results
boot_summary = {
    'models': {name: {
        'AUC': round(results[name]['AUC'], 4),
        'CI_lo': round(bootstrap_results[name]['CI_lo'], 4),
        'CI_hi': round(bootstrap_results[name]['CI_hi'], 4)
    } for name in model_order},
    'conformal_bootstrap': {
        'marginal_coverage': round(float(marg_in_set.mean()), 4),
        'marginal_CI': [round(marg_ci[0], 4), round(marg_ci[1], 4)],
        'cc_aki_coverage': round(float(cc_in_set[y_ext_arr==1].mean()), 4),
        'cc_aki_CI': [round(cc_aki_ci[0], 4), round(cc_aki_ci[1], 4)]
    },
    'best_model': best_name,
    'n_bootstrap': n_boot
}
with open(os.path.join(RES, 'bootstrap_results.json'), 'w') as f:
    json.dump(boot_summary, f, indent=2)
print("  bootstrap_results.json saved")

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
print(f"Best model: {best_name} (AUC={results[best_name]['AUC']:.3f})")
print(f"MLP external AUC: {results['MLP']['AUC']:.3f}")
