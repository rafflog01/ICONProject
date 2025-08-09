# Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Config (niente salvataggi su disco)

RANDOM_STATE = 42
FAST_MODE = True  # True = ricerca più veloce; False = più approfondita (più lenta)
RECALL_MIN = 0.70  # banda recall: minimo
RECALL_MAX = 0.75  # banda recall: massimo

# 1) Caricamento dataset

try:
    dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")

print(dataset.info())

# 2) Target e features

y = dataset['Overall Survival Status'].str.upper().map({
    '0:LIVING': 0, '1:DECEASED': 1, 'ALIVE': 0, 'DEAD': 1
})

X = dataset.drop([
    "Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
    "Tumor Tissue Origin", "Last Communication Contact", "Patient's Vital Status",
    "Oncotree Code", "Overall Survival Status"
], axis=1)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 3) Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# 4) Pipeline + RandomizedSearchCV

pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=0.0)),
    ("sampler", SMOTE(random_state=RANDOM_STATE)),  # placeholder, sostituito via search
    ("scaler", StandardScaler(with_mean=False)),
    ("pca", PCA(random_state=RANDOM_STATE)),
    ("svc", SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=RANDOM_STATE))
])

if FAST_MODE:
    param_distributions = {
        "var__threshold": [0.0, 0.001],
        "sampler": [
            SMOTE(random_state=RANDOM_STATE),
            BorderlineSMOTE(random_state=RANDOM_STATE)
        ],
        "pca__n_components": [None, 0.95],
        "svc__C": [0.5, 1, 5, 10, 50, 100],
        "svc__gamma": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    }
    n_iter = 30
    cv_folds = 3
else:
    param_distributions = {
        "var__threshold": [0.0, 0.0005, 0.001, 0.01],
        "sampler": [
            SMOTE(random_state=RANDOM_STATE),
            BorderlineSMOTE(random_state=RANDOM_STATE)
        ],
        "pca__n_components": [None, 0.99, 0.95],
        "svc__C": [0.1, 0.5, 1, 5, 10, 50, 100, 200, 500],
        "svc__gamma": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1]
    }
    n_iter = 60
    cv_folds = 5

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "f1": "f1",
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision"
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring=scoring,
    refit="average_precision",  # ottimizza PR-AUC
    cv=cv,
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE
)

print("\n[INFO] Avvio RandomizedSearchCV...")
search.fit(X_train, y_train)

print("\n=== MIGLIORI PARAMETRI (refit su average_precision) ===")
print(search.best_params_)
print("\n=== SCORE CV DEL MIGLIOR MODELLO ===")
for k in scoring.keys():
    print(f"{k}: {search.cv_results_[f'mean_test_{k}'][search.best_index_]:.4f}")

best_model = search.best_estimator_

# 5) Calibrazione (Platt/sigmoid)

X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
)

# Fit del migliore modello sul sotto-train
best_model.fit(X_tr_sub, y_tr_sub)

# Calibrazione
calibrator = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv='prefit')
calibrator.fit(X_val, y_val)

# Probabilità calibrate su validation
val_probs = calibrator.predict_proba(X_val)[:, 1]

# Curva PR e soglie
prec, rec, thr = precision_recall_curve(y_val, val_probs)  # thr ha len = len(prec)-1 = len(rec)-1

# Indici in banda di recall
rec_in_thr = rec[1:]  # recall allineato con thr
idx_band = np.where((rec_in_thr >= RECALL_MIN) & (rec_in_thr <= RECALL_MAX))[0]

if len(idx_band) > 0:
    band_prec = prec[idx_band + 1]  # precision allineata a thr
    best_idx = idx_band[int(np.argmax(band_prec))]
    chosen_thr = thr[best_idx]
    chosen_prec = prec[best_idx + 1]
    chosen_rec = rec[best_idx + 1]
    chosen_rule = f"max precision nella banda [{RECALL_MIN:.2f}, {RECALL_MAX:.2f}]"
else:
    idx_ok = np.where(rec_in_thr >= RECALL_MIN)[0]
    if len(idx_ok) > 0:
        ok_prec = prec[idx_ok + 1]
        best_idx = idx_ok[int(np.argmax(ok_prec))]
        chosen_thr = thr[best_idx]
        chosen_prec = prec[best_idx + 1]
        chosen_rec = rec[best_idx + 1]
        chosen_rule = f"max precision con recall >= {RECALL_MIN:.2f}"
    else:
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        best_idx_f1 = int(np.nanargmax(f1s))
        chosen_thr = thr[best_idx_f1] if best_idx_f1 < len(thr) else 0.5
        chosen_prec = prec[best_idx_f1]
        chosen_rec = rec[best_idx_f1]
        chosen_rule = "max F1 (fallback)"

print(f"\n[CALIBRAZIONE + SOGLIA ROBUSTA]")
print(f"Regola: {chosen_rule}")
print(f"Soglia calibrata: {chosen_thr:.4f}")
print(f"Precision (val): {chosen_prec:.4f} | Recall (val): {chosen_rec:.4f}")
print(f"PR-AUC (validation, calibrata): {average_precision_score(y_val, val_probs):.4f}")

# 6) Valutazione finale su TEST con modello calibrato + soglia robusta

best_model.fit(X_train, y_train)
calibrator_final = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv='prefit')
calibrator_final.fit(X_val, y_val)

test_probs = calibrator_final.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= chosen_thr).astype(int)

acc = accuracy_score(y_test, test_pred)
bacc = balanced_accuracy_score(y_test, test_pred)
f1_bin = f1_score(y_test, test_pred)
f1_mac = f1_score(y_test, test_pred, average='macro')
roc = roc_auc_score(y_test, test_probs)
ap = average_precision_score(y_test, test_probs)
cm = confusion_matrix(y_test, test_pred)

print("\n=== TEST @ soglia robusta (calibrata) – senza salvataggi ===")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"F1 (binary): {f1_bin:.4f}")
print(f"F1 (macro): {f1_mac:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC (AP): {ap:.4f}")
print("\nClassification Report:\n", classification_report(y_test, test_pred))
print("Confusion Matrix:\n", cm)

# Matrice di confusione normalizzata
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(5, 4))
sn.heatmap(pd.DataFrame(cm_norm,
                        index=['Classe 0', 'Classe 1'],
                        columns=['Pred 0', 'Pred 1']),
           annot=True, fmt='.2f', cmap='Blues')
plt.title(f'Confusion Matrix (%) - Test @ soglia robusta ({RECALL_MIN:.2f}–{RECALL_MAX:.2f})')
plt.ylabel('Reale')
plt.xlabel('Predetto')
plt.tight_layout()
plt.show()

# Curva PR (test) e punto operativo

prec_t, rec_t, thr_t = precision_recall_curve(y_test, test_probs)
plt.figure(figsize=(6, 5))
plt.step(rec_t, prec_t, where='post', color='b', alpha=0.8, label='PR (calibrata)')
if len(thr_t) > 0:
    idx_thr = np.argmin(np.abs(thr_t - chosen_thr))  # allineato a prec[1:], rec[1:]
    plt.scatter(rec_t[idx_thr + 1], prec_t[idx_thr + 1], c='red', s=50, label='Operating point')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'PR Curve - Test (AP={average_precision_score(y_test, test_probs):.3f})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 8) Curva ROC (test)

fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_score(y_test, test_probs):.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 9) Deviazione standard via CV

skf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Usiamo l'estimatore "best_model" (pipeline completa) per evitare leakage; metrica: accuracy
cv_scores = cross_val_score(best_model, X, y, cv=skf_full, scoring='accuracy', n_jobs=-1)
print("\n[CV STRATIFICATA - tutto il dataset] ")
print(f"Media accuracy: {np.mean(cv_scores):.4f}")
print(f"Deviazione standard: {np.std(cv_scores):.4f}")
print(f"Varianza: {np.var(cv_scores):.4f}")

# Grafico semplice varianza/dev std
fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(['variance', 'std dev'], [np.var(cv_scores), np.std(cv_scores)], color=['#5DA5DA', '#60BD68'])
for i, v in enumerate([np.var(cv_scores), np.std(cv_scores)]):
    ax.text(i, v + max(1e-3, v) * 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
ax.set_title('Stabilità CV (accuracy)')
plt.tight_layout()
plt.show()
