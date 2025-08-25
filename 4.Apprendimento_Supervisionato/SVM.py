"""
@autore: Raffaele Loglisci
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from inspect import signature
import time

# Caricamento dataset
try:
    dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")

print(dataset.info())

# Target e features
y = dataset['Overall Survival Status'].str.upper().map({
    '0:LIVING': 0, '1:DECEASED': 1, 'ALIVE': 0, 'DEAD': 1
})

X = dataset.drop([
    "Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
    "Tumor Tissue Origin", "Last Communication Contact", "Patient's Vital Status",
    "Oncotree Code", "Overall Survival Status"
], axis=1)

# One-Hot Encoding per categoriche
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Repeated Holdout (outer) + 5-fold CV (inner)
n_runs = 10
test_size = 0.20
seeds = list(range(42, 42 + n_runs))
plot_each_run = True

acc_list, f1_list, auc_list, ap_list = [], [], [], []


# Pipeline per il tuning interno (imputazione + SMOTE + scaling + SVC)
def make_pipeline(seed):
    return ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=seed)),
        ("varth", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=False, random_state=seed, cache_size=2000))
    ])


# Griglia iperparametri SVM
param_grid = [
    # kernel RBF
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.5, 1, 5, 10],
        "svc__gamma": ["scale", 1e-3],
    },
    # kernel lineare: C limitati (quelli alti sono molto lenti)
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.5, 1, 2],
    }
]

for i, seed in enumerate(seeds, start=1):
    # Outer split 80/20 (stratificato)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y
    )

    # Inner 5-fold CV sul SOLO training per tuning (no leakage)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = make_pipeline(seed)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_inner,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=1
    )

    print(f"\n[Run {i}/{n_runs}] Avvio GridSearchCV su {len(X_train)} campioni, {X_train.shape[1]} feature...")
    t0 = time.time()
    grid.fit(X_train, y_train)
    t1 = time.time()
    print(f"[Run {i}/{n_runs}] GridSearchCV completata in {t1 - t0:.1f}s")

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    # Valutazione sul test dell'outer split
    y_pred = best_model.predict(X_test)

    # Punteggi continui per ROC/AP
    if hasattr(best_model, "decision_function"):
        scores = best_model.decision_function(X_test)
        if isinstance(scores, np.ndarray) and scores.ndim > 1 and scores.shape[1] > 1:
            scores = scores[:, 1]
    elif hasattr(best_model, "predict_proba"):
        scores = best_model.predict_proba(X_test)[:, 1]
    else:
        scores = y_pred

    acc = accuracy_score(y_test, y_pred)
    f1v = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, scores)
    except ValueError:
        auc = np.nan
    ap = average_precision_score(y_test, scores)

    acc_list.append(acc)
    f1_list.append(f1v)
    auc_list.append(auc)
    ap_list.append(ap)

    print(f"[Run {i}/{n_runs} - seed={seed}] "
          f"CV(5-fold) best Acc={best_cv_score:.4f} | best_params={best_params} | "
          f"Test Acc={acc:.4f} F1={f1v:.4f} AUC={auc:.4f} AP={ap:.4f}")
    print("\nClassification report (test):\n", classification_report(y_test, y_pred))

    # Grafici
    if plot_each_run:
        # Matrice di confusione normalizzata
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Matrice di confusione normalizzata (%) - Test (Run {i})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        # ROC Curve
        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(y_test, scores)
            plt.figure(figsize=(6, 5))
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Test (Run {i})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, scores)
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.figure(figsize=(6, 5))
        plt.step(recall, precision, color='b', alpha=0.8, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall (AP={ap:.3f}) - Test (Run {i})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# Riepilogo statistico (outer test)
def _summ(arr):
    arr = np.array(arr, dtype=float)
    return np.nanmean(arr), np.nanstd(arr), np.nanvar(arr)


acc_mean, acc_std, acc_var = _summ(acc_list)
f1_mean, f1_std, f1_var = _summ(f1_list)
auc_mean, auc_std, auc_var = _summ(auc_list)
ap_mean, ap_std, ap_var = _summ(ap_list)

print("\n=== Riepilogo Repeated Holdout (outer test) con inner 5-fold CV ===")
print(f"Accuracy:          mean={acc_mean:.4f} | std={acc_std:.4f} | var={acc_var:.6f}")
print(f"F1:                mean={f1_mean:.4f} | std={f1_std:.4f} | var={f1_var:.6f}")
print(f"ROC-AUC:           mean={auc_mean:.4f} | std={auc_std:.4f} | var={auc_var:.6f}")
print(f"Average Precision: mean={ap_mean:.4f} | std={ap_std:.4f} | var={ap_var:.6f}")

# Tabella riassuntiva
tabella = pd.DataFrame([
    {"metrica": "accuracy", "mean": acc_mean, "std": acc_std, "var": acc_var},
    {"metrica": "f1", "mean": f1_mean, "std": f1_std, "var": f1_var},
    {"metrica": "roc_auc", "mean": auc_mean, "std": auc_std, "var": auc_var},
    {"metrica": "avg_prec", "mean": ap_mean, "std": ap_std, "var": ap_var},
], columns=["metrica", "mean", "std", "var"])

print("\nTabella riassuntiva (metriche su test, media/std/var):")
print(tabella.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Grafici riepilogativi
# Boxplot delle metriche su test
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

metrics_data = {
    "Accuracy": acc_list,
    "F1": f1_list,
    "ROC-AUC": auc_list,
    "Average Precision": ap_list
}

for ax, (name, values) in zip(axes, metrics_data.items()):
    vals = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    ax.boxplot(vals, vert=True, patch_artist=True, boxprops=dict(facecolor='#5DA5DA'))
    ax.set_title(f'{name} – distribuzione su {n_runs} run')
    ax.set_ylabel(name)

plt.tight_layout()
plt.show()

# Grafico varianza e deviazione standard (accuracy)
fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(['variance', 'std dev'], [acc_var, acc_std], color=['#5DA5DA', '#60BD68'])
for i, v in enumerate([acc_var, acc_std]):
    ax.text(i, v + max(1e-3, v) * 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
ax.set_title('Stabilità (Accuracy) – varianza e std su test')
plt.tight_layout()
plt.show()
