
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from inspect import signature

# Caricamento del dataset
try:
    dataset = pd.read_csv("../2.Ontologia/breast-cancer.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("2.Ontologia/breast-cancer.csv")
    except FileNotFoundError:
        try:
            dataset = pd.read_csv("../../2.Ontologia/breast-cancer.csv")
        except FileNotFoundError:
            dataset = pd.read_csv("breast-cancer.csv")

# Preprocessing
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
print(dataset.info())

# Preparazione dati
y = dataset['diagnosis']
X = dataset.drop(['id', 'diagnosis'], axis=1)
X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Divisione train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grid Search per gamma
param_grid = {'gamma': [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]}
svm_model = svm.SVC(kernel='rbf', probability=True)
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Stampa risultati Grid Search
print("Risultati Grid Search:")
for gamma, mean_score in zip(grid_search.cv_results_['param_gamma'],
                             grid_search.cv_results_['mean_test_score']):
    print(f"Gamma: {gamma:<10}Mean Score: {mean_score:.4f}")
print(f"\nMiglior valore di gamma: {grid_search.best_params_['gamma']}")
print(f"Miglior score: {grid_search.best_score_:.4f}")

# Training con il miglior parametro
clf = svm.SVC(kernel='rbf', gamma=grid_search.best_params_['gamma'], probability=True)
clf.fit(X_train_scaled, y_train)

# Predizioni
prediction = clf.predict(X_test_scaled)
accuracy = accuracy_score(prediction, y_test)
print(f'\nAccuracy Score: {accuracy:.4f}')

# Report di classificazione
print('\nClassification Report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n', confusion_matrix(y_test, prediction))

# Matrice di confusione
conf_matrix = confusion_matrix(y_test, prediction)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 7))
sn.heatmap(pd.DataFrame(conf_matrix_percent,
                        index=['Benigno (0)', 'Maligno (1)'],
                        columns=['Pred Benigno (0)', 'Pred Maligno (1)']),
           annot=True, fmt='.2f', cmap='Oranges')
plt.title('Matrice di Confusione Normalizzata (%)')
plt.ylabel('Valore Reale')
plt.xlabel('Predizione')
plt.show()

# Cross-validation
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
print('\nRisultati Cross-validation:')
print(f'Media accuracy: {np.mean(cv_scores):.4f}')
print(f'Deviazione standard: {np.std(cv_scores):.4f}')
print(f'Varianza: {np.var(cv_scores):.4f}')

# Curva ROC
probs = clf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FP RATE')
plt.ylabel('TP RATE')
plt.show()

# Curva Precision-Recall
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

data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values, color='orange')
plt.show()
