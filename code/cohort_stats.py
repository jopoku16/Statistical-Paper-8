import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\19565\Desktop\Statistical Paper 8\data\sepsis_icu_synthetic.csv")
has_ckd = df["chronic_kidney_disease"] == 1
cr = df["creatinine"].copy()
aki = pd.Series(0, index=df.index)
aki[(~has_ckd) & (cr >= 2.5)] = 1
aki[(has_ckd) & (cr >= 3.5)] = 1
aki[cr.isna()] = np.nan
df["aki"] = aki
df = df.dropna(subset=["aki"])
df["aki"] = df["aki"].astype(int)

noaki = df[df["aki"] == 0]
yesaki = df[df["aki"] == 1]

def iqr(s):
    return f"{s.median():.0f} [{s.quantile(0.25):.0f}-{s.quantile(0.75):.0f}]"

def iqr1(s):
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}]"

print(f"N: {len(noaki)} / {len(yesaki)}")
print(f"Age: {iqr(noaki['age'])} / {iqr(yesaki['age'])}")
male_noaki = (noaki["gender"] == "M").mean() * 100
male_aki = (yesaki["gender"] == "M").mean() * 100
print(f"Male%: {male_noaki:.1f} / {male_aki:.1f}")
print(f"CKD%: {noaki['chronic_kidney_disease'].mean()*100:.1f} / {yesaki['chronic_kidney_disease'].mean()*100:.1f}")
print(f"SOFA: {iqr(noaki['sofa_score'])} / {iqr(yesaki['sofa_score'])}")
print(f"APACHE: {iqr1(noaki['apache_iv'])} / {iqr1(yesaki['apache_iv'])}")
print(f"Lactate: {iqr1(noaki['lactate_mmol'])} / {iqr1(yesaki['lactate_mmol'])}")
print(f"Vasopressors%: {noaki['vasopressors_flag'].mean()*100:.1f} / {yesaki['vasopressors_flag'].mean()*100:.1f}")
