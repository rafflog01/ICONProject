import numpy as np
import pandas as pd
import seaborn as sn
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_recall_curve, average_precision_score,
                             roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Caricamento dataset breast_msk_2018
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

# Pulizia colonne inutili
discard_cols = [
    "Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
    "Tumor Tissue Origin", "Last Communication Contact", "Oncotree Code", "Overall Survival Status"
]
keep_cols = [c for c in dataset.columns if c not in discard_cols]
dataset = dataset[keep_cols]

# Label
if "Patient's Vital Status" not in dataset.columns:
    raise ValueError("Colonna 'Patient's Vital Status' mancante nel dataset!")

y = dataset["Patient's Vital Status"].map({'Alive': 1, 'Deceased': 0})
X = dataset.drop(["Patient's Vital Status"], axis=1)

# Codifica di colonne categoriche secondo mappa clinica
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

X = X.select_dtypes(include=['float64', 'int64'])

# Split stratificato train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

imputer = SimpleImputer(strategy='mean')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# y_train resta invariato (serie)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imp, y_train.reset_index(drop=True))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
pd.Series(y_train).value_counts().plot(kind='bar', ax=axs[0], color='skyblue')
axs[0].set_title("Prima di SMOTE")
axs[0].set_xlabel("Classe")
axs[0].set_ylabel("Frequenza")
pd.Series(y_train_bal).value_counts().plot(kind='bar', ax=axs[1], color='orange')
axs[1].set_title("Dopo SMOTE")
axs[1].set_xlabel("Classe")
plt.tight_layout()
plt.show()

X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scelta K ottimale sui dati bilanciati
error = []
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test_imp)

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_bal_scaled, y_train_bal)
    pred_i = knn.predict(X_test_scaled)
    error.append(np.mean(pred_i != y_test))

optimal_k = error.index(min(error)) + 1
print(f"\nK ottimale trovato: {optimal_k}")

# Cross-validation su training bilanciato
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=optimal_k))
])
cv_scores = cross_val_score(
    pipeline, X_train_bal, y_train_bal, cv=5
)

print('\nCross-validation results (SMOTE train):')
print(f'Mean accuracy: {np.mean(cv_scores):.4f}')
print(f'Standard deviation: {np.std(cv_scores):.4f}')
print(f'Variance: {np.var(cv_scores):.5f}')

# Training KNN definitivo su tutto il train bilanciato
pipeline.fit(X_train_bal, y_train_bal)
prediction = pipeline.predict(X_test_imp)

accuracy = accuracy_score(y_test, prediction)
print(f"\nAccuracy Score (test set): {accuracy:.4f}")
print('\nClassification Report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n', confusion_matrix(y_test, prediction))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='black', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.xticks(range(1, 20, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

conf_matrix = confusion_matrix(y_test, prediction)
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

# ROC Curve
if hasattr(pipeline.named_steps['knn'], "predict_proba"):
    probs = pipeline.predict_proba(X_test_imp)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    print('\nAUC: %.3f' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Precision Recall Curve
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

# F1 Score
f1 = f1_score(y_test, prediction)
print(f'\nF1 Score: {f1:.4f}')

# Varianza e std cross-validation
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
plt.figure(figsize=(6, 3))
plt.bar(list(data.keys()), list(data.values()), color='orange')
plt.title('Varianza e Deviazione Standard Cross-validation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pipeline per scaling + knn
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Definizione della griglia degli iperparametri
param_grid = {
    'knn__n_neighbors': list(range(1, 21)),
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2],  # 1=Manhattan, 2=Euclidea
}

# Istanzia il grid search con validazione incrociata a 5 fold
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Esegui grid search solo sul training bilanciato
grid.fit(X_train_bal, y_train_bal)

print(f"Migliori iperparametri trovati: {grid.best_params_}")
print(f"Accuracy media in cross-validation: {grid.best_score_:.4f}")

# Ottieni il miglior modello trovato e valutalo sul test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_imp)
print("\nCLASSIFICATION REPORT SUL TEST SET:")
print(classification_report(y_test, y_pred))
