import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===============================
# 1) Caricamento dataset
# ===============================
try:
    dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        try:
            dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")
        except FileNotFoundError:
            dataset = pd.read_csv("../../2.Ontologia/breast_msk_2018_clinical_data.csv")

print(dataset.info())

# ===============================
# 2) Pulizia colonne inutili
# ===============================
discard_cols = [
    "Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
    "Tumor Tissue Origin", "Last Communication Contact", "Oncotree Code", "Overall Survival Status"
]
keep_cols = [c for c in dataset.columns if c not in discard_cols]
dataset = dataset[keep_cols]

# ===============================
# 3) Label e features
# ===============================
if "Patient's Vital Status" not in dataset.columns:
    raise ValueError("Colonna 'Patient's Vital Status' mancante nel dataset!")

y = dataset["Patient's Vital Status"].map({'Alive': 1, 'Deceased': 0})
X = dataset.drop(["Patient's Vital Status"], axis=1)

# Mappatura di alcune categoriche cliniche
categorical_columns_to_map = {
    "ER Status of Sequenced Sample": {"Positive": 1, "Negative": 0},
    "ER Status of the Primary": {"Positive": 1, "Negative": 0},
    "PR Status of Sequenced Sample": {"Positive": 1, "Negative": 0},
    "PR Status of the Primary": {"Positive": 1, "Negative": 0},
    "HER2 Primary Status": {"Positive": 1, "Negative": 0, "Equivocal": -1, "Unk/ND": -2},
    "Menopausal Status At Diagnosis": {"Pre": 0, "Peri": 1, "Post": 2, "Unknown": -1},
}

for col, mapping in categorical_columns_to_map.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

# Teniamo solo numeriche (le categoriche non mappate verranno ignorate)
X = X.select_dtypes(include=['float64', 'int64'])

# ===============================
# 4) Train/Test split (stratificato)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# ===============================
# 5) Pipeline + GridSearchCV (tuning senza leakage)
#    Tutti i passi (imputazione, SMOTE, scaling, KNN) sono nella Pipeline
# ===============================
pipe = ImbPipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": list(range(1, 21)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],  # 1 = Manhattan, 2 = Euclidea
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Nota: puoi cambiare 'scoring' (es. 'f1', 'roc_auc', 'balanced_accuracy') se dataset sbilanciato.
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# Fit SOLO sul train (la Pipeline gestisce tutto internamente)
grid.fit(X_train, y_train)

print(f"\nMigliori iperparametri: {grid.best_params_}")
print(f"Accuracy media in CV (GridSearchCV): {grid.best_score_:.4f}")

# ===============================
# 6) Valutazione finale sul test
# ===============================
best_model = grid.best_estimator_

# Predizione sul test "grezzo": la Pipeline gestisce imputazione/SMOTE/scaling internamente
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n[Test] Accuracy: {accuracy:.4f}")
print("\nClassification Report (test):")
print(classification_report(y_test, y_pred))

# Matrice di confusione (percentuali per riga)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(10, 7))
sn.heatmap(
    pd.DataFrame(conf_matrix_percent,
                 index=['Deceased (0)', 'Alive (1)'],
                 columns=['Pred Deceased (0)', 'Pred Alive (1)']),
    annot=True, fmt='.2f', cmap='Oranges'
)
plt.title('Matrice di Confusione Normalizzata (%) - Test')
plt.ylabel('Valore Reale')
plt.xlabel('Predizione')
plt.show()

# ===============================
# 7) Curve ROC e Precision-Recall (se disponibili)
# ===============================
if hasattr(best_model.named_steps['knn'], "predict_proba"):
    probs = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    print(f"\nAUC ROC (test): {auc:.3f}")

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    average_precision = average_precision_score(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='red', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='orange', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AP = {average_precision:.3f}) - Test')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# ===============================
# 8) F1 finale (complementare ad accuracy)
# ===============================
f1 = f1_score(y_test, y_pred)
print(f"\nF1 (test): {f1:.4f}")

# ===============================
# 9) 5-fold CV senza leakage (Pipeline migliore clonata per fold)
#    Metriche per fold + media/std + grafico varianza/std (accuracy)
# ===============================
print("\n=== 5-fold CV senza leakage (Pipeline migliore): metriche per fold ===")

acc_folds, f1_folds, roc_folds, ap_folds = [], [], [], []
fold_idx = 1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Clona la Pipeline migliore (imputazione, SMOTE, scaling, KNN)
    model_fold = clone(best_model)

    # Fit SOLO sul train della fold (tutti i passi della pipeline rifatti per fold)
    model_fold.fit(X_tr, y_tr)

    # Predizioni classi e probabilità
    y_pred_fold = model_fold.predict(X_val)
    if hasattr(model_fold, "predict_proba"):
        scores = model_fold.predict_proba(X_val)[:, 1]
    else:
        # fallback (meno ideale per ROC/AP)
        scores = y_pred_fold

    # Metriche per fold
    acc = accuracy_score(y_val, y_pred_fold)
    f1v = f1_score(y_val, y_pred_fold)
    try:
        roc = roc_auc_score(y_val, scores)
    except ValueError:
        roc = np.nan  # raro caso di una sola classe in validation
    ap = average_precision_score(y_val, scores)

    acc_folds.append(acc)
    f1_folds.append(f1v)
    roc_folds.append(roc)
    ap_folds.append(ap)

    print(f"[Fold {fold_idx}] Acc={acc:.4f} | F1={f1v:.4f} | ROC-AUC={roc:.4f} | AP={ap:.4f}")
    fold_idx += 1

# Riepilogo per-fold
def _summ(arr):
    arr = np.array(arr, dtype=float)
    return arr, np.nanmean(arr), np.nanstd(arr)

acc_arr, acc_mean, acc_std = _summ(acc_folds)
f1_arr,  f1_mean,  f1_std  = _summ(f1_folds)
roc_arr, roc_mean, roc_std = _summ(roc_folds)
ap_arr,  ap_mean,  ap_std  = _summ(ap_folds)

print("\n=== Riepilogo 5-fold (valori per fold + media/std) ===")
print(f"Accuracy:          {np.round(acc_arr, 4)} | mean={acc_mean:.4f} | std={acc_std:.4f}")
print(f"F1:                {np.round(f1_arr, 4)}  | mean={f1_mean:.4f}  | std={f1_std:.4f}")
print(f"ROC-AUC:           {np.round(roc_arr, 4)} | mean={roc_mean:.4f} | std={roc_std:.4f}")
print(f"Average Precision: {np.round(ap_arr, 4)}  | mean={ap_mean:.4f}  | std={ap_std:.4f}")

# Tabella per documentazione
tabella = pd.DataFrame([
    {"metrica": "accuracy", "fold_1": acc_arr[0], "fold_2": acc_arr[1], "fold_3": acc_arr[2], "fold_4": acc_arr[3], "fold_5": acc_arr[4], "media": acc_mean, "std": acc_std},
    {"metrica": "f1",       "fold_1": f1_arr[0],  "fold_2": f1_arr[1],  "fold_3": f1_arr[2],  "fold_4": f1_arr[3],  "fold_5": f1_arr[4],  "media": f1_mean,  "std": f1_std},
    {"metrica": "roc_auc",  "fold_1": roc_arr[0], "fold_2": roc_arr[1], "fold_3": roc_arr[2], "fold_4": roc_arr[3], "fold_5": roc_arr[4], "media": roc_mean, "std": roc_std},
    {"metrica": "avg_prec", "fold_1": ap_arr[0],  "fold_2": ap_arr[1],  "fold_3": ap_arr[2],  "fold_4": ap_arr[3],  "fold_5": ap_arr[4],  "media": ap_mean,  "std": ap_std},
], columns=["metrica","fold_1","fold_2","fold_3","fold_4","fold_5","media","std"])

print("\nTabella riassuntiva (da copiare in documentazione):")
print(tabella.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# tabella.to_csv("cv_metrics_knn.csv", index=False)  # opzionale

# Grafico varianza e deviazione standard (accuracy su 5 fold)
var_acc = np.nanvar(acc_arr)
std_acc = np.nanstd(acc_arr)

fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(['variance', 'std dev'], [var_acc, std_acc], color=['#5DA5DA', '#60BD68'])
for i, v in enumerate([var_acc, std_acc]):
    ax.text(i, v + max(1e-3, v) * 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
ax.set_title('Stabilità CV (accuracy) – varianza e std')
plt.tight_layout()
plt.show()