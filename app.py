# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import joblib
# import datetime as dt
# import sqlite3
# import os


# from flask import Flask, render_template, request, jsonify
# import os


# # ===============================
# # 1. Custom wrapper class
# #    MUST be defined before joblib.load
# # ===============================
# class HybridSuperEnsemble:
#     """
#     Wrapper class as used during training.
#     Holds:
#       - scaler
#       - encoder (from AE)
#       - xgb_model
#       - ada_model
#       - cnn_model
#       - rnn_model
#     """

#     def __init__(self, scaler, encoder, xgb_model, ada_model, cnn_model, rnn_model):
#         self.scaler = scaler
#         self.encoder = encoder
#         self.xgb_model = xgb_model
#         self.ada_model = ada_model
#         self.cnn_model = cnn_model
#         self.rnn_model = rnn_model

#     def _forward(self, X_2d):
#         # scale
#         X_s = self.scaler.transform(X_2d)
#         # AE latent
#         Z = self.encoder.predict(X_s, verbose=0)
#         # fused features for tree models
#         F = np.hstack([X_s, Z])

#         # tree model probs
#         p_xgb = self.xgb_model.predict_proba(F)[:, 1]
#         p_ada = self.ada_model.predict_proba(F)[:, 1]

#         # DL models – sequence input
#         n_features = X_s.shape[1]
#         X_dl = X_s.reshape(-1, n_features, 1)
#         p_cnn = self.cnn_model.predict(X_dl, verbose=0).ravel()
#         p_rnn = self.rnn_model.predict(X_dl, verbose=0).ravel()

#         # ensemble
#         p_ens = (p_xgb + p_ada + p_cnn + p_rnn) / 4.0
#         return p_ens

#     def predict_proba(self, X_2d):
#         p_mal = self._forward(X_2d)
#         p_mal = np.clip(p_mal, 1e-7, 1 - 1e-7)
#         p_ben = 1.0 - p_mal
#         return np.vstack([p_ben, p_mal]).T

#     def predict(self, X_2d, threshold=0.5):
#         p_mal = self._forward(X_2d)
#         return (p_mal >= threshold).astype(int)


# # ===============================
# # 2. Flask app + DB setup
# # ===============================
# app = Flask(__name__)

# # paths
# MODEL_PATH = "hybrid_super_ensemble_model.pkl"
# DB_PATH = "pred_logs.db"

# # load model
# hybrid_model: HybridSuperEnsemble = joblib.load(MODEL_PATH)

# # WDBC features (30)
# FEATURES = [
#     "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
#     "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
#     "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
#     "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
#     "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
#     "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
# ]


# PLOT_DIR = "static/plots"

# PLOT_FILES = {
#     "boxplot": "boxplots_top_features.png",
#     "hist_prob": "hist_malignant_prob.png",
#     "corr": "corr_heatmap.png",
#     "acc_f1": "benchmark_acc_f1.png",
#     "cm_hybrid": "cm_hybrid.png",
#     "roc_all": "roc_curves_all.png",
#     "shap": "shap_summary_xgb.png"
# }




# @app.route("/visualize")
# def visualize():
#     plots = {}
#     for key, fname in PLOT_FILES.items():
#         path = os.path.join(PLOT_DIR, fname)
#         if os.path.exists(path):
#             plots[key] = "/" + path.replace("\\", "/")
#     return render_template("visualize.html", plots=plots)


# @app.route("/benchmark")
# def benchmark():
#     bench_path = "data/benchmark_results.csv"
#     if not os.path.exists(bench_path):
#         return render_template("benchmark.html", table=None)

#     df = pd.read_csv(bench_path)
#     return render_template("benchmark.html", table=df.to_dict(orient="records"))


# @app.route("/xai")
# def xai():
#     shap_path = os.path.join(PLOT_DIR, "shap_summary_xgb.png")
#     shap_url = "/" + shap_path.replace("\\", "/") if os.path.exists(shap_path) else None
#     return render_template("xai.html", shap_plot=shap_url)

# # ---------- SQLite helpers ----------
# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute(
#         """
#         CREATE TABLE IF NOT EXISTS predictions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             time TEXT,
#             source TEXT,
#             label TEXT,
#             prob REAL,
#             extra TEXT
#         )
#         """
#     )
#     conn.commit()
#     conn.close()


# def log_prediction(source, label, prob, extra=""):
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute(
#         "INSERT INTO predictions (time, source, label, prob, extra) VALUES (?, ?, ?, ?, ?)",
#         (
#             dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             source, label, float(prob), extra
#         )
#     )
#     conn.commit()
#     conn.close()


# def get_recent_predictions(limit=50):
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute(
#         "SELECT time, source, label, prob, extra FROM predictions ORDER BY id DESC LIMIT ?",
#         (limit,)
#     )
#     rows = cur.fetchall()
#     conn.close()
#     return [
#         {"time": r[0], "source": r[1], "label": r[2], "prob": r[3], "extra": r[4]}
#         for r in rows
#     ]


# def get_stats():
#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()
#     cur.execute("SELECT COUNT(*), SUM(label='Malignant'), AVG(prob) FROM predictions")
#     total, malignant, avg_prob = cur.fetchone()
#     conn.close()
#     malignant = malignant or 0
#     benign = (total or 0) - malignant
#     return {
#         "total": total or 0,
#         "malignant": malignant,
#         "benign": benign,
#         "avg_prob": round(avg_prob, 4) if avg_prob is not None else None
#     }


# # initialize DB once
# if not os.path.exists(DB_PATH):
#     init_db()
# else:
#     # ensure table exists
#     init_db()


# # ---------- Hybrid prediction helper ----------
# def predict_hybrid(X_2d):
#     probs = hybrid_model.predict_proba(X_2d)[:, 1]
#     preds = hybrid_model.predict(X_2d)
#     return preds, probs


# # ===============================
# # 3. Routes
# # ===============================

# @app.route("/", methods=["GET"])
# def index():
#     history = get_recent_predictions(limit=20)
#     return render_template("index.html", features=FEATURES, history=history)


# @app.route("/predict_manual", methods=["POST"])
# def predict_manual():
#     try:
#         vals = [float(request.form.get(f)) for f in FEATURES]
#         X_input = np.array(vals).reshape(1, -1)

#         preds, probs = predict_hybrid(X_input)
#         label = "Malignant" if preds[0] == 1 else "Benign"
#         prob = float(probs[0])

#         log_prediction("Manual", label, prob)

#         history = get_recent_predictions(limit=20)
#         return render_template(
#             "index.html",
#             features=FEATURES,
#             manual_result={"label": label, "prob": prob},
#             history=history
#         )
#     except Exception as e:
#         history = get_recent_predictions(limit=20)
#         return render_template(
#             "index.html",
#             features=FEATURES,
#             error=f"Manual prediction error: {str(e)}",
#             history=history
#         )


# @app.route("/predict_csv", methods=["POST"])
# def predict_csv():
#     try:
#         file = request.files.get("file")
#         if not file or file.filename == "":
#             raise ValueError("No CSV file uploaded.")

#         df = pd.read_csv(file)

#         # ----- CSV validation -----
#         missing = [c for c in FEATURES if c not in df.columns]
#         if missing and df.shape[1] < 30:
#             raise ValueError(
#                 "CSV must contain at least 30 feature columns. Missing: " + ", ".join(missing)
#             )

#         # prepare X
#         has_label = False
#         if "diagnosis" in df.columns:
#             has_label = True
#             y_true = df["diagnosis"].map({"B": 0, "M": 1}).values
#             X = df[FEATURES].values
#         else:
#             # allow both exact-feature name CSV + 'no header' template
#             if set(FEATURES).issubset(df.columns):
#                 X = df[FEATURES].values
#             else:
#                 X = df.iloc[:, :30].values  # assume correct order

#         preds, probs = predict_hybrid(X)

#         results = []
#         malignant_count = int((preds == 1).sum())
#         benign_count = len(preds) - malignant_count

#         metrics = None
#         if has_label:
#             from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#             acc = accuracy_score(y_true, preds)
#             prec = precision_score(y_true, preds)
#             rec = recall_score(y_true, preds)
#             f1 = f1_score(y_true, preds)
#             metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

#         for i in range(len(preds)):
#             label = "Malignant" if preds[i] == 1 else "Benign"
#             prob = float(probs[i])
#             row_info = {"row": i+1, "label": label, "prob": prob}

#             if has_label:
#                 true_label = "Malignant" if y_true[i] == 1 else "Benign"
#                 correct = (label == true_label)
#                 row_info["true_label"] = true_label
#                 row_info["correct"] = correct

#             results.append(row_info)
#             log_prediction("CSV", label, prob, extra=f"row {i+1}")

#         history = get_recent_predictions(limit=20)

#         return render_template(
#             "index.html",
#             features=FEATURES,
#             csv_results=results,
#             csv_summary={
#                 "total": len(preds),
#                 "malignant": malignant_count,
#                 "benign": benign_count
#             },
#             csv_metrics=metrics,
#             history=history
#         )

#     except Exception as e:
#         history = get_recent_predictions(limit=20)
#         return render_template(
#             "index.html",
#             features=FEATURES,
#             error=f"CSV prediction error: {str(e)}",
#             history=history
#         )


# @app.route("/dashboard")
# def dashboard():
#     stats = get_stats()
#     history = get_recent_predictions(limit=100)
#     return render_template("dashboard.html", stats=stats, history=history)


# # -------- Simple JSON API --------
# @app.route("/api/predict", methods=["POST"])
# def api_predict():
#     """
#     JSON body:
#     {
#         "features": [30 values]   OR
#         "batch": [[30 vals], [30 vals], ...]
#     }
#     """
#     data = request.get_json(force=True)

#     if "batch" in data:
#         X = np.array(data["batch"], dtype=float)
#     elif "features" in data:
#         X = np.array(data["features"], dtype=float).reshape(1, -1)
#     else:
#         return jsonify({"error": "Provide 'features' or 'batch'"}), 400

#     preds, probs = predict_hybrid(X)
#     labels = ["Malignant" if p == 1 else "Benign" for p in preds]
#     return jsonify({
#         "labels": labels,
#         "probs_malignant": probs.tolist()
#     })


# if __name__ == "__main__":
#     app.run(port=5000, debug=True)


from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import datetime as dt
import os
from pymongo import MongoClient


# ===============================
# 1. Custom wrapper class
#    MUST be defined before joblib.load
# ===============================
class HybridSuperEnsemble:
    """
    Wrapper class as used during training.
    Holds:
      - scaler
      - encoder (from AE)
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
        # scale
        X_s = self.scaler.transform(X_2d)

        # AE latent
        Z = self.encoder.predict(X_s, verbose=0)

        # fused features for tree models
        F = np.hstack([X_s, Z])

        # tree model probs
        p_xgb = self.xgb_model.predict_proba(F)[:, 1]
        p_ada = self.ada_model.predict_proba(F)[:, 1]

        # DL models – sequence input
        n_features = X_s.shape[1]
        X_dl = X_s.reshape(-1, n_features, 1)
        p_cnn = self.cnn_model.predict(X_dl, verbose=0).ravel()
        p_rnn = self.rnn_model.predict(X_dl, verbose=0).ravel()

        # ensemble
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


# ===============================
# 2. Flask app + MongoDB setup
# ===============================
app = Flask(__name__)

# paths
MODEL_PATH = "hybrid_super_ensemble_model.pkl"

# load model
hybrid_model: HybridSuperEnsemble = joblib.load(MODEL_PATH)

# MongoDB config
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "bc_app"
MONGO_COLLECTION = "predictions"

client = MongoClient(MONGO_URI)
mongo_db = client[MONGO_DB_NAME]
pred_col = mongo_db[MONGO_COLLECTION]

# WDBC features (30)
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

PLOT_DIR = "static/plots"

PLOT_FILES = {
    "boxplot": "boxplots_top_features.png",
    "hist_prob": "hist_malignant_prob.png",
    "corr": "corr_heatmap.png",
    "acc_f1": "benchmark_acc_f1.png",
    "cm_hybrid": "cm_hybrid.png",
    "roc_all": "roc_curves_all.png",
    "shap": "shap_summary_xgb.png"
}


@app.route("/visualize")
def visualize():
    plots = {}
    for key, fname in PLOT_FILES.items():
        path = os.path.join(PLOT_DIR, fname)
        if os.path.exists(path):
            plots[key] = "/" + path.replace("\\", "/")
    return render_template("visualize.html", plots=plots)


@app.route("/benchmark")
def benchmark():
    bench_path = "data/benchmark_results.csv"
    if not os.path.exists(bench_path):
        return render_template("benchmark.html", table=None)

    df = pd.read_csv(bench_path)
    return render_template("benchmark.html", table=df.to_dict(orient="records"))


@app.route("/xai")
def xai():
    shap_path = os.path.join(PLOT_DIR, "shap_summary_xgb.png")
    shap_url = "/" + shap_path.replace("\\", "/") if os.path.exists(shap_path) else None
    return render_template("xai.html", shap_plot=shap_url)


# ---------- MongoDB helpers ----------
def init_db():
    pred_col.create_index("time")


def log_prediction(source, label, prob, extra=""):
    doc = {
        "time": dt.datetime.now(),
        "source": source,
        "label": label,
        "prob": float(prob),
        "extra": extra
    }
    pred_col.insert_one(doc)


def get_recent_predictions(limit=50):
    rows = pred_col.find({}, {"_id": 0}).sort("time", -1).limit(limit)
    history = []

    for r in rows:
        history.append({
            "time": r["time"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r.get("time"), dt.datetime) else str(r.get("time", "")),
            "source": r.get("source", ""),
            "label": r.get("label", ""),
            "prob": r.get("prob", 0),
            "extra": r.get("extra", "")
        })

    return history


def get_stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total": {"$sum": 1},
                "malignant": {
                    "$sum": {
                        "$cond": [{"$eq": ["$label", "Malignant"]}, 1, 0]
                    }
                },
                "avg_prob": {"$avg": "$prob"}
            }
        }
    ]

    res = list(pred_col.aggregate(pipeline))

    if not res:
        return {
            "total": 0,
            "malignant": 0,
            "benign": 0,
            "avg_prob": None
        }

    total = res[0].get("total", 0)
    malignant = res[0].get("malignant", 0)
    benign = total - malignant
    avg_prob = res[0].get("avg_prob", None)

    return {
        "total": total,
        "malignant": malignant,
        "benign": benign,
        "avg_prob": round(avg_prob, 4) if avg_prob is not None else None
    }


# initialize MongoDB
init_db()


# ---------- Hybrid prediction helper ----------
def predict_hybrid(X_2d):
    probs = hybrid_model.predict_proba(X_2d)[:, 1]
    preds = hybrid_model.predict(X_2d)
    return preds, probs


# ===============================
# 3. Routes
# ===============================

@app.route("/", methods=["GET"])
def index():
    history = get_recent_predictions(limit=20)
    return render_template("index.html", features=FEATURES, history=history)


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    try:
        vals = [float(request.form.get(f)) for f in FEATURES]
        X_input = np.array(vals).reshape(1, -1)

        preds, probs = predict_hybrid(X_input)
        label = "Malignant" if preds[0] == 1 else "Benign"
        prob = float(probs[0])

        log_prediction("Manual", label, prob)

        history = get_recent_predictions(limit=20)
        return render_template(
            "index.html",
            features=FEATURES,
            manual_result={"label": label, "prob": prob},
            history=history
        )
    except Exception as e:
        history = get_recent_predictions(limit=20)
        return render_template(
            "index.html",
            features=FEATURES,
            error=f"Manual prediction error: {str(e)}",
            history=history
        )


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            raise ValueError("No CSV file uploaded.")

        df = pd.read_csv(file)

        # ----- CSV validation -----
        missing = [c for c in FEATURES if c not in df.columns]
        if missing and df.shape[1] < 30:
            raise ValueError(
                "CSV must contain at least 30 feature columns. Missing: " + ", ".join(missing)
            )

        # prepare X
        has_label = False
        if "diagnosis" in df.columns:
            has_label = True
            y_true = df["diagnosis"].map({"B": 0, "M": 1}).values
            X = df[FEATURES].values
        else:
            if set(FEATURES).issubset(df.columns):
                X = df[FEATURES].values
            else:
                X = df.iloc[:, :30].values

        preds, probs = predict_hybrid(X)

        results = []
        malignant_count = int((preds == 1).sum())
        benign_count = len(preds) - malignant_count

        metrics = None
        if has_label:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds)
            rec = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)
            metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

        for i in range(len(preds)):
            label = "Malignant" if preds[i] == 1 else "Benign"
            prob = float(probs[i])
            row_info = {"row": i + 1, "label": label, "prob": prob}

            if has_label:
                true_label = "Malignant" if y_true[i] == 1 else "Benign"
                correct = (label == true_label)
                row_info["true_label"] = true_label
                row_info["correct"] = correct

            results.append(row_info)
            log_prediction("CSV", label, prob, extra=f"row {i+1}")

        history = get_recent_predictions(limit=20)

        return render_template(
            "index.html",
            features=FEATURES,
            csv_results=results,
            csv_summary={
                "total": len(preds),
                "malignant": malignant_count,
                "benign": benign_count
            },
            csv_metrics=metrics,
            history=history
        )

    except Exception as e:
        history = get_recent_predictions(limit=20)
        return render_template(
            "index.html",
            features=FEATURES,
            error=f"CSV prediction error: {str(e)}",
            history=history
        )


@app.route("/dashboard")
def dashboard():
    stats = get_stats()
    history = get_recent_predictions(limit=100)
    return render_template("dashboard.html", stats=stats, history=history)


# -------- Simple JSON API --------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON body:
    {
        "features": [30 values]
        OR
        "batch": [[30 vals], [30 vals], ...]
    }
    """
    data = request.get_json(force=True)

    if "batch" in data:
        X = np.array(data["batch"], dtype=float)
    elif "features" in data:
        X = np.array(data["features"], dtype=float).reshape(1, -1)
    else:
        return jsonify({"error": "Provide 'features' or 'batch'"}), 400

    preds, probs = predict_hybrid(X)
    labels = ["Malignant" if p == 1 else "Benign" for p in preds]

    return jsonify({
        "labels": labels,
        "probs_malignant": probs.tolist()
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)