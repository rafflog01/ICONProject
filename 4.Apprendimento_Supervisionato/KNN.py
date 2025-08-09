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
# 5) Pipeline + GridSearchCV (solo questo per il tuning)
#    Tutti i passi (imputazione, SMOTE, scaling) sono dentro la Pipeline,
#    così la CV è metodologicamente corretta e senza leakage.
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

# Nota: puoi cambiare 'scoring' in 'f1', 'roc_auc' o 'balanced_accuracy' se c'è sbilanciamento.
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# Fit SOLO sul train originale (non bilanciato/scalato/imputato fuori dalla Pipeline)
grid.fit(X_train, y_train)

print(f"\nMigliori iperparametri: {grid.best_params_}")
print(f"Accuracy media in CV: {grid.best_score_:.4f}")

# ===============================
# 6) Valutazione finale sul test
# ===============================
best_model = grid.best_estimator_

# Predizione sul test "grezzo": la Pipeline gestisce imputazione/scaling internamente
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy sul test: {accuracy:.4f}")
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
plt.title('Matrice di Confusione Normalizzata (%)')
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
    print(f"\nAUC ROC: {auc:.3f}")

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    average_precision = average_precision_score(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='red', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='orange', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AP = {average_precision:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# ===============================
# 8) F1 finale (complementare ad accuracy)
# ===============================
f1 = f1_score(y_test, y_pred)
print(f"\nF1 Score (test): {f1:.4f}")