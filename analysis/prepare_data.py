# analysis/prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- 1. Load & rename full WDBC ----------
raw_path = "../data/wdbc.data"
df = pd.read_csv(raw_path, header=None)

# col0 = id, col1 = diagnosis, col2-31 = 30 features
df = df[[1] + list(range(2, 32))]

feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractall_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se",
    "concavity_se","concave points_se","symmetry_se","fractall_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractall_dimension_worst"
]

df.columns = ["diagnosis"] + feature_names

# Save full realistic dataset (C)
df.to_csv("../data/wdbc_realistic_full_renamed.csv", index=False)

print("Saved: wdbc_realistic_full_renamed.csv (full 569 rows)")

# ---------- 2. Balanced 300 (A) ----------
m = df[df.diagnosis == "M"]
b = df[df.diagnosis == "B"]

n = 150  # 150 M + 150 B = 300
m300 = m.sample(n, random_state=42)
b300 = b.sample(n, random_state=42)

balanced300 = pd.concat([m300, b300]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced300.to_csv("../data/wdbc_balanced_300_renamed.csv", index=False)

print("Saved: wdbc_balanced_300_renamed.csv (150 M + 150 B)")

# ---------- 3. Balanced Full (B) ----------
min_class = min(len(m), len(b))
m_full = m.sample(min_class, random_state=42)
b_full = b.sample(min_class, random_state=42)

balanced_full = pd.concat([m_full, b_full]).sample(frac=1, random_state=42).reset_index(drop=True)
balanced_full.to_csv("../data/wdbc_balanced_full_renamed.csv", index=False)

print(f"Saved: wdbc_balanced_full_renamed.csv ({min_class} M + {min_class} B)")

# ---------- 4. Stratified Train/Test split 80/20 (D) ----------
# Use full realistic dataset
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].map({"B": 0, "M": 1}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

train_df = pd.DataFrame(X_train, columns=feature_names)
train_df.insert(0, "diagnosis", ["M" if v == 1 else "B" for v in y_train])

test_df = pd.DataFrame(X_test, columns=feature_names)
test_df.insert(0, "diagnosis", ["M" if v == 1 else "B" for v in y_test])

train_df.to_csv("../data/wdbc_train_80.csv", index=False)
test_df.to_csv("../data/wdbc_test_20.csv", index=False)

print("Saved: wdbc_train_80.csv & wdbc_test_20.csv (stratified 80/20)")
