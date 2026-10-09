# Breast Cancer Prediction Using Hybrid Machine Learning and Explainable AI

A Flask web application for classifying breast tumors as **benign** or **malignant** using a hybrid machine learning and deep learning ensemble, with SHAP-based interpretation of its XGBoost component.

## Project Overview

This academic project combines predictive modeling, explainable AI, and a web interface using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The ensemble integrates XGBoost, AdaBoost, a convolutional neural network (CNN), and a recurrent neural network (RNN). An autoencoder generates latent features that are combined with standardized input features for the tree-based models.

Users can enter tumor measurements manually, upload a CSV file for batch prediction, review prediction history, and explore model evaluation plots.

## Key Features

- **Binary classification:** Predict benign or malignant tumor labels and malignant-class probabilities.
- **Hybrid ensemble:** Combine predictions from XGBoost, AdaBoost, CNN, and RNN models.
- **Feature representation:** Use autoencoder-derived latent features alongside standardized measurements.
- **Manual and batch prediction:** Submit individual samples through a form or multiple samples through CSV upload.
- **Prediction API:** Accept single-sample and batch JSON requests.
- **Explainable AI:** Display a SHAP summary plot for the XGBoost component.
- **Visualization pages:** Explore feature distributions, correlations, confusion matrices, ROC curves, and model comparisons.
- **Prediction history:** Store manual and CSV predictions in MongoDB.
- **Dashboard:** Summarize prediction counts, class distribution, and average malignant probability.
- **Data preparation:** Generate renamed datasets, balanced subsets, and a stratified train/test split.

## Machine Learning Approach

The inference pipeline follows these steps:

1. Standardize the 30 input features using the fitted scaler.
2. Generate latent representations using the trained autoencoder's encoder.
3. Concatenate standardized and latent features for XGBoost and AdaBoost.
4. Reshape standardized features for CNN and RNN inference.
5. Average the malignant-class probabilities from the four classifiers.
6. Assign the malignant label when the averaged probability is at least `0.5`.

The training notebook includes a 1D CNN and a bidirectional LSTM/GRU-based RNN, autoencoder reconstruction-based outlier filtering, and stratified cross-validation experiments.

The autoencoder supplies feature representations; the final classification probability is the average of the **four classifiers**.

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

| Property | Description |
| --- | --- |
| Samples | 569 |
| Benign samples | 357 |
| Malignant samples | 212 |
| Input features | 30 numerical measurements |
| Original labels | `B` = Benign, `M` = Malignant |
| Internal encoding | `0` = Benign, `1` = Malignant |

The features describe cell nucleus characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each characteristic has mean, standard error, and worst-value measurements.

The data preparation script:

- Removes the identifier column and assigns feature names.
- Creates a renamed version of the original dataset.
- Generates a balanced 300-sample subset with 150 samples per class.
- Generates a balanced 424-sample subset with 212 samples per class.
- Creates a stratified 80/20 split containing 455 training and 114 test samples using `random_state=42`.

These inputs are numerical cell morphology measurements, rather than genomic or gene-expression data.

## Explainable AI

The analysis script uses **SHAP (SHapley Additive exPlanations)** with `shap.TreeExplainer` to interpret the XGBoost component.

The summary plot shows feature contributions across standardized original features and autoencoder-derived latent features. The Flask application's XAI page displays this generated plot.

**Explainability scope:** The current implementation provides a global explanation of the XGBoost component. It does not generate individual SHAP explanations for each web prediction or explain the complete averaged ensemble.

## Technologies Used

| Area | Technologies |
| --- | --- |
| Programming | Python |
| Machine learning | Scikit-learn, XGBoost |
| Deep learning | TensorFlow / Keras |
| Explainability | SHAP |
| Data processing | NumPy, Pandas |
| Web application | Flask, HTML, CSS, JavaScript |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Active database | MongoDB, PyMongo |

The archive also contains an earlier SQLite prediction database and commented SQLite code. The active application uses MongoDB.

## Project Structure

| Path | Purpose |
| --- | --- |
| `app.py` | Flask routes, ensemble inference, and MongoDB prediction logging |
| `BC_prediction_(1) (3).ipynb` | Training experiments, cross-validation, and model export |
| `hybrid_super_ensemble_model.pkl` | Ensemble artifact loaded by the Flask application |
| `hybrid_super_scaler.pkl` | Separately exported scaler artifact |
| `model/` | Model and scaler copies used by the analysis script |
| `analysis/prepare_data.py` | Dataset preparation and stratified splitting |
| `analysis/generate_plots_and_shap.py` | Benchmark evaluation, visualizations, and SHAP generation |
| `data/` | Raw data, processed datasets, train/test files, and benchmark results |
| `static/css/` | Application styles |
| `static/images/` | Background imagery |
| `static/plots/` | Generated plots |
| `templates/` | Prediction, dashboard, visualization, benchmark, and XAI pages |
| `requirements.txt` | Python dependency list |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Elahi-01/Breast-cancer-prediction.git
cd Breast-cancer-prediction
```

Alternatively, extract the project archive and open a terminal in the folder containing `app.py`.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pymongo
```

The supplied requirements pin TensorFlow to `2.18.0`. PyMongo is imported by the active application but is absent from the supplied dependency list, so it must be installed separately or added to `requirements.txt`.

### 4. Configure MongoDB

Install and start MongoDB before launching the application. The current settings in `app.py` use:

| Setting | Value |
| --- | --- |
| Connection URI | `mongodb://localhost:27017/` |
| Database | `bc_app` |
| Collection | `predictions` |

Update the MongoDB settings in `app.py` if your database uses a different connection.

### 5. Verify model files

The Flask application expects `hybrid_super_ensemble_model.pkl` in the project root. The analysis script expects the ensemble artifact inside `model/`.

Keep these copies consistent when replacing or retraining the model. The ensemble artifact contains its fitted scaler and encoder.

## Running the Application

Run this command from the project root:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

| Route | Purpose |
| --- | --- |
| `/` | Manual input, CSV upload, and recent prediction history |
| `/dashboard` | Prediction statistics and history |
| `/visualize` | Generated plots |
| `/benchmark` | Saved model comparison metrics |
| `/xai` | XGBoost SHAP summary plot |
| `POST /api/predict` | Single-sample or batch JSON prediction |

Manual and CSV submissions are logged to MongoDB. The JSON prediction API returns results without writing prediction history.

## Input Format

Each sample must contain the **30 numerical features** in the order defined by `FEATURES` in `app.py`.

For CSV uploads:

- Use the exact feature names expected by `app.py`.
- Include one sample per row.
- Optionally include a `diagnosis` column containing `B` or `M` to calculate accuracy, precision, recall, and F1-score on the uploaded data.
- Exclude identifier columns from positional input.

**Feature-name compatibility:** The data preparation script and supplied processed CSVs use `fractall_dimension_*`, while the active application expects `fractal_dimension_*`. Rename those three columns before uploading a labeled CSV or standardize the names in the preparation script and regenerate the files. Preserve the spaces in the `concave points_*` feature names.

For the JSON API, send either:

- `{"features": [...]}` with exactly 30 numerical values; or
- `{"batch": [[...], [...]]}` with 30 numerical values in each row.

The response includes `labels` and `probs_malignant`.

## Generating Data, Plots, and SHAP Analysis

The analysis scripts use paths relative to the `analysis/` directory. Run them from that directory:

```bash
cd analysis
python prepare_data.py
python generate_plots_and_shap.py
```

Outputs include processed CSV datasets, `data/benchmark_results.csv`, and plots in `static/plots/`.

## Model Evaluation

The analysis pipeline reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curves

The hybrid ensemble is compared with Logistic Regression, SVM, Random Forest, KNN, AdaBoost, Gradient Boosting, and XGBoost.

The notebook includes cross-validation experiments and a final model export trained on the full dataset. Before presenting the saved benchmark values as independent test performance, verify that the evaluated model artifact was trained without the test samples. The train/test CSV split alone does not establish this separation.

Serialized models also depend on compatible library versions. Most dependencies in the supplied requirements are unpinned, so record the training environment when reproducing or updating the artifacts.

## Research Applications

This project demonstrates:

- Machine learning for breast tumor classification.
- Hybrid ensembles combining tree-based and neural network models.
- Autoencoder-based feature representation.
- Explainable AI for biomedical classification.
- Flask deployment of a predictive modeling workflow.

This is an academic and research prototype. Its outputs are not a clinical diagnosis, and clinical use would require independent validation.

## Author

**MD Fazley Elahi**

GitHub: [Elahi-01](https://github.com/Elahi-01)

Repository: [Breast-cancer-prediction](https://github.com/Elahi-01/Breast-cancer-prediction)

## License

No license file is included in the supplied project archive. Add a `LICENSE` file to define permissions for reuse and distribution.
