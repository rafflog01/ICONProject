"""
@autore: Raffaele Loglisci
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from inspect import signature

# ===============================
# 1) Caricamento del dataset
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
# 2) Target e features
# ===============================
y = dataset['Overall Survival Status'].str.upper().map({
    '0:LIVING': 0, '1:DECEASED': 1, 'ALIVE': 0, 'DEAD': 1
})

X = dataset.drop([
    "Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
    "Tumor Tissue Origin", "Last Communication Contact", "Patient's Vital Status", "Oncotree Code",
    "Overall Survival Status"
], axis=1)

# One-Hot Encoding delle categoriche
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ===============================
# 3) Split train/test (holdout)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# ===============================
# 4) Imputazione + SMOTE + (opz.) Scaling
# ===============================
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Oversampling SMOTE solo sul train
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imp, y_train)

# RF non richiede scaling, ma lo includiamo per coerenza con pipeline
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test_imp)

# ===============================
# 5) Grid Search su Pipeline (scaler + RF)
# ===============================
pipe = Pipeline([
    ('scaler', StandardScaler()),  # non strettamente necessario per RF
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 7, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__bootstrap': [True, False],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_bal_scaled, y_train_bal)

print("Migliori iperparametri trovati:", grid.best_params_)
print(f"Accuracy media in cross-validation (GridSearchCV): {grid.best_score_:.4f}")

# ===============================
# 6) Valutazione sul test set con il miglior modello
# ===============================
best_rf = grid.best_estimator_

prediction = best_rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, prediction)
print(f'\n[Test] Accuracy: {accuracy:.4f}')
print('\nClassification report (test):\n', classification_report(y_test, prediction))
print('\nConfusion matrix (test):\n', confusion_matrix(y_test, prediction))

# Matrice di confusione normalizzata
conf_matrix = confusion_matrix(y_test, prediction)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
plt.title('Matrice di confusione normalizzata (%) - Test')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC-AUC e PR-AUC sul test
probs = best_rf.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f'AUC (test): {auc:.3f}')

average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve (AP={average_precision:0.2f}) - Test')
plt.show()

f1 = f1_score(y_test, prediction)
print('\nF1 (test): ', f1)

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FP RATE')
plt.ylabel('TP RATE')
plt.title('ROC Curve - Test')
plt.show()

# ===============================
# 7) 5-fold CV senza leakage con metriche per fold
#    (imputazione, SMOTE e scaling rifatti in ogni fold)
# ===============================
print("\n=== 5-fold CV senza leakage: metriche per fold ===")

# Prendiamo gli iperparametri migliori del classificatore RF
best_rf_params = grid.best_estimator_.named_steps['rf'].get_params()
# Evita duplicazione del parametro random_state nel costruttore
best_rf_params = {k: v for k, v in best_rf_params.items() if k != 'random_state'}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_folds, f1_folds, roc_folds, ap_folds = [], [], [], []
fold_idx = 1

for tr_idx, val_idx in cv.split(X, y):
    # Split grezzo (dopo OHE già effettuato su X completo)
    X_tr_raw, X_val_raw = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Imputazione (fit sul train; transform su val)
    imputer_cv = SimpleImputer(strategy="median")
    X_tr_imp = imputer_cv.fit_transform(X_tr_raw)
    X_val_imp = imputer_cv.transform(X_val_raw)

    # SMOTE SOLO sul train
    smote_cv = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = smote_cv.fit_resample(X_tr_imp, y_tr)

    # Scaling (fit sul train bilanciato; applica a val)
    scaler_cv = StandardScaler()
    X_tr_bal_scaled = scaler_cv.fit_transform(X_tr_bal)
    X_val_scaled = scaler_cv.transform(X_val_imp)

    # Modello RF con i migliori iperparametri (random_state impostato qui)
    rf_fold = RandomForestClassifier(random_state=42, **best_rf_params)
    rf_fold.fit(X_tr_bal_scaled, y_tr_bal)

    # Predizioni su validation della fold
    probs_fold = rf_fold.predict_proba(X_val_scaled)[:, 1]
    y_pred_fold = (probs_fold >= 0.5).astype(int)

    # Metriche per fold
    acc = accuracy_score(y_val, y_pred_fold)
    f1v = f1_score(y_val, y_pred_fold)
    try:
        roc = roc_auc_score(y_val, probs_fold)
    except ValueError:
        roc = np.nan  # raro caso di una sola classe in val
    ap = average_precision_score(y_val, probs_fold)

    acc_folds.append(acc)
    f1_folds.append(f1v)
    roc_folds.append(roc)
    ap_folds.append(ap)

    print(f"[Fold {fold_idx}] Acc={acc:.4f} | F1={f1v:.4f} | ROC-AUC={roc:.4f} | AP={ap:.4f}")
    fold_idx += 1

# Riepilogo CV (nan-safe per ROC-AUC)
def _summ(arr):
    arr = np.array(arr, dtype=float)
    return arr, np.nanmean(arr), np.nanstd(arr)

acc_arr, acc_mean, acc_std = _summ(acc_folds)
f1_arr, f1_mean, f1_std = _summ(f1_folds)
roc_arr, roc_mean, roc_std = _summ(roc_folds)
ap_arr, ap_mean, ap_std = _summ(ap_folds)

print("\n=== Riepilogo 5-fold (valori per fold + media/std) ===")
print(f"Accuracy:          {np.round(acc_arr, 4)} | mean={acc_mean:.4f} | std={acc_std:.4f}")
print(f"F1:                {np.round(f1_arr, 4)} | mean={f1_mean:.4f} | std={f1_std:.4f}")
print(f"ROC-AUC:           {np.round(roc_arr, 4)} | mean={roc_mean:.4f} | std={roc_std:.4f}")
print(f"Average Precision: {np.round(ap_arr, 4)} | mean={ap_mean:.4f} | std={ap_std:.4f}")

# Tabella per documentazione
tabella = pd.DataFrame([
    {"metrica": "accuracy", "fold_1": acc_arr[0], "fold_2": acc_arr[1], "fold_3": acc_arr[2], "fold_4": acc_arr[3], "fold_5": acc_arr[4], "media": acc_mean, "std": acc_std},
    {"metrica": "f1",       "fold_1": f1_arr[0],  "fold_2": f1_arr[1],  "fold_3": f1_arr[2],  "fold_4": f1_arr[3],  "fold_5": f1_arr[4],  "media": f1_mean,  "std": f1_std},
    {"metrica": "roc_auc",  "fold_1": roc_arr[0], "fold_2": roc_arr[1], "fold_3": roc_arr[2], "fold_4": roc_arr[3], "fold_5": roc_arr[4], "media": roc_mean, "std": roc_std},
    {"metrica": "avg_prec", "fold_1": ap_arr[0],  "fold_2": ap_arr[1],  "fold_3": ap_arr[2],  "fold_4": ap_arr[3],  "fold_5": ap_arr[4],  "media": ap_mean,  "std": ap_std},
], columns=["metrica","fold_1","fold_2","fold_3","fold_4","fold_5","media","std"])

print("\nTabella riassuntiva (da copiare in documentazione):")
print(tabella.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
# tabella.to_csv("cv_metrics_randomforest.csv", index=False)  # opzionale

# ===============================
# Grafico varianza e deviazione standard (accuracy su 5 fold)
# ===============================
var_acc = np.nanvar(acc_arr)
std_acc = np.nanstd(acc_arr)

fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(['variance', 'std dev'], [var_acc, std_acc], color=['#5DA5DA', '#60BD68'])
for i, v in enumerate([var_acc, std_acc]):
    ax.text(i, v + max(1e-3, v) * 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
ax.set_title('Stabilità CV (accuracy) – varianza e std')
plt.tight_layout()
plt.show()

