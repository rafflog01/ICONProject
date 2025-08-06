"""
@autore: Raffaele Loglisci
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from inspect import signature
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Caricamento del dataset
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

# Target
y = dataset['Overall Survival Status'].str.upper().map({'0:LIVING': 0, '1:DECEASED': 1, 'ALIVE': 0, 'DEAD': 1})

X = dataset.drop(["Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
                  "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
                  "Tumor Tissue Origin", "Last Communication Contact", "Patient's Vital Status", "Oncotree Code",
                  "Overall Survival Status"
                  ], axis=1)

# Encoding delle variabili categoriche
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# Imputazione valori mancanti
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Oversampling SMOTE solo sul train
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

#GRID SEARCH
pipe = Pipeline([
    ('scaler', StandardScaler()),
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
grid.fit(X_train_bal, y_train_bal)

print("Migliori iperparametri trovati:", grid.best_params_)
print(f"Accuracy media in cross-validation: {grid.best_score_:.4f}")

#Valutazione sul test set con il miglior modello
best_rf = grid.best_estimator_
prediction = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f'\nAccuracy score (test set): {accuracy:.4f}')
print('\nClassification report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n', confusion_matrix(y_test, prediction))

# Matrice di confusione normalizzata
conf_matrix = confusion_matrix(y_test, prediction)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
plt.title('Matrice di confusione normalizzata')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

best_rf = grid.best_estimator_

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Esegue la cross-validation stratificata
cv_scores = cross_val_score(best_rf, X, y, cv=skf, scoring='accuracy')

print('\n[CV STRATIFICATA - tutto il dataset]')
print('Media: ', np.mean(cv_scores))
print('Dev standard: ', np.std(cv_scores))
print('Varianza: ', np.var(cv_scores))

# ROC-AUC e altre metriche
probs = best_rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f'AUC: {auc:.3f}')

average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f}')
plt.show()

f1 = f1_score(y_test, prediction)
print('\nf1 score: ', f1)

data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FP RATE')
plt.ylabel('TP RATE')
plt.show()
