# analysis/generate_plots_and_shap.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



# =====================================================
# 1. HybridSuperEnsemble class (same as in app.py)
# =====================================================

class HybridSuperEnsemble:
    """
    Wrapper class:
      - scaler
      - encoder (AE)
      - xgb_model
      - ada_model
      - cnn_model
      - rnn_model
    """

    def __init__(self, scaler, encoder, xgb_model, ada_model, cnn_model, rnn_model):
        self.scaler = scaler
        self.encoder = encoder
        self.xgb_model = xgb_model
        self.ada_model = ada_model
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model

    def _forward(self, X_2d):
        # 1) scale
        X_s = self.scaler.transform(X_2d)
        # 2) AE latent
        Z = self.encoder.predict(X_s, verbose=0)
        # 3) fused features for tree models
        F = np.hstack([X_s, Z])

        # 4) boosting probs
        p_xgb = self.xgb_model.predict_proba(F)[:, 1]
        p_ada = self.ada_model.predict_proba(F)[:, 1]

        # 5) DL models – sequence input
        n_features = X_s.shape[1]
        X_dl = X_s.reshape(-1, n_features, 1)
        p_cnn = self.cnn_model.predict(X_dl, verbose=0).ravel()
        p_rnn = self.rnn_model.predict(X_dl, verbose=0).ravel()

        # 6) ensemble average
        p_ens = (p_xgb + p_ada + p_cnn + p_rnn) / 4.0
        return p_ens

    def predict_proba(self, X_2d):
        p_mal = self._forward(X_2d)
        p_mal = np.clip(p_mal, 1e-7, 1 - 1e-7)
        p_ben = 1.0 - p_mal
        return np.vstack([p_ben, p_mal]).T

    def predict(self, X_2d, threshold=0.5):
        p_mal = self._forward(X_2d)
        return (p_mal >= threshold).astype(int)


# =====================================================
# 2. Paths
# =====================================================

DATA_TRAIN = "../data/wdbc_train_80.csv"
DATA_TEST  = "../data/wdbc_test_20.csv"
MODEL_PATH = "../model/hybrid_super_ensemble_model.pkl"
PLOT_DIR   = "../static/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# =====================================================
# 3. Load data
# =====================================================

train_df = pd.read_csv(DATA_TRAIN)
test_df  = pd.read_csv(DATA_TEST)

feature_cols = [c for c in train_df.columns if c != "diagnosis"]

X_train = train_df[feature_cols].values
y_train = train_df["diagnosis"].map({"B": 0, "M": 1}).values

X_test = test_df[feature_cols].values
y_test = test_df["diagnosis"].map({"B": 0, "M": 1}).values

# =====================================================
# 4. Load hybrid model
# =====================================================

hybrid_model: HybridSuperEnsemble = joblib.load(MODEL_PATH)

# =====================================================
# 5. E – Thesis Visualizations
# =====================================================

# E1. Boxplot – top 5 features
plt.figure(figsize=(10, 6))
melt_df = train_df.melt(
    id_vars="diagnosis",
    value_vars=feature_cols[:5],
    var_name="feature",
    value_name="value"
)
sns.boxplot(data=melt_df, x="feature", y="value", hue="diagnosis")
plt.xticks(rotation=45, ha="right")
plt.title("Top 5 Features Distribution by Diagnosis")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplots_top_features.png"))
plt.close()

# E2. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = train_df[feature_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "corr_heatmap.png"))
plt.close()

# E3. Hybrid probabilities distribution on test set
probs_hybrid = hybrid_model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 5))
sns.histplot(probs_hybrid, bins=20, kde=True)
plt.title("Hybrid Model Probability Distribution (Test set)")
plt.xlabel("Probability")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "hist_malignant_prob.png"))
plt.close()

print("Saved: boxplots_top_features.png, corr_heatmap.png, hist_malignant_prob.png")

# =====================================================
# 6. F – Benchmark ML models (LR, SVM, RF, KNN, Ada, XGB vs Hybrid)
#     + Accuracy/F1 barplot
#     + Confusion Matrix (Hybrid)
#     + ROC curves (all models)
# =====================================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN (k=9, distance)": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=400,
        learning_rate=1.5,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    ),
}

results = []
all_probs = {}   # ROC curve এর জন্য সব model-এর prob এখানে রাখব

for name, clf in models.items():
    clf.fit(X_train_s, y_train)
    prob = clf.predict_proba(X_test_s)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    auc  = roc_auc_score(y_test, prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    })
    all_probs[name] = prob

# Hybrid model metrics (same as before)
hyb_pred = (probs_hybrid >= 0.5).astype(int)
hyb_acc  = accuracy_score(y_test, hyb_pred)
hyb_prec = precision_score(y_test, hyb_pred)
hyb_rec  = recall_score(y_test, hyb_pred)
hyb_f1   = f1_score(y_test, hyb_pred)
hyb_auc  = roc_auc_score(y_test, probs_hybrid)

results.append({
    "Model": "Hybrid Super Ensemble",
    "Accuracy": hyb_acc,
    "Precision": hyb_prec,
    "Recall": hyb_rec,
    "F1": hyb_f1,
    "ROC_AUC": hyb_auc
})
all_probs["Hybrid Super Ensemble"] = probs_hybrid

# ---------- Save benchmark table ----------
bench_df = pd.DataFrame(results)
bench_df.to_csv("../data/benchmark_results.csv", index=False)
print("Saved: benchmark_results.csv")

# ---------- 6a. Accuracy vs F1 bar plot ----------
model_names = [r["Model"] for r in results]
acc_values  = [r["Accuracy"] for r in results]
f1_values   = [r["F1"] for r in results]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, acc_values, width, label="Accuracy")
plt.bar(x + width/2, f1_values, width, label="F1-score")
plt.xticks(x, model_names, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Model Comparison – Accuracy & F1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "benchmark_acc_f1.png"), dpi=150)
plt.close()
print("Saved: benchmark_acc_f1.png")

# ---------- 6b. Confusion Matrix – Hybrid model ----------
cm = confusion_matrix(y_test, hyb_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign (0)", "Malignant (1)"]
)
plt.figure(figsize=(5, 5))
disp.plot(values_format="d")
plt.title("Confusion Matrix – Hybrid Super Ensemble")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "cm_hybrid.png"), dpi=150)
plt.close()
print("Saved: cm_hybrid.png")

# ---------- 6c. ROC curves – all models ----------
plt.figure(figsize=(8, 6))

for r in results:
    name = r["Model"]
    prob = all_probs[name]
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['ROC_AUC']:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")  # random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – All Models")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curves_all.png"), dpi=150)
plt.close()
print("Saved: roc_curves_all.png")



# =====================================================
# 7. G – SHAP Explainability for XGBoost component
# =====================================================

# 7.1: Fused features বানানো (exactly like training time)
# scaler এবং encoder আমরা hybrid_model-এর ভেতর থেকেই নেবো
X_train_s_for_xgb = hybrid_model.scaler.transform(X_train)
Z_train = hybrid_model.encoder.predict(X_train_s_for_xgb, verbose=0)

F_train = np.hstack([X_train_s_for_xgb, Z_train])   # fused features

xgb_model = hybrid_model.xgb_model

# 7.2: SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(F_train)

# কিছু shap version এ shap_values list হয় (প্রতিটা class এর জন্য আলাদা)
if isinstance(shap_values, list):
    shap_vals_to_plot = shap_values[1]   # class 1 = Malignant
else:
    shap_vals_to_plot = shap_values

# 7.3: feature name list = original 30 + latent features
latent_dim = Z_train.shape[1]
all_feature_names = feature_cols + [f"latent_{i+1}" for i in range(latent_dim)]

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals_to_plot,
    F_train,
    feature_names=all_feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "shap_summary_xgb.png"),
            bbox_inches="tight", dpi=150)
plt.close()

print("Saved: shap_summary_xgb.png")
print("✅ All plots & benchmark generated successfully!")
