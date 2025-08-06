"""
@autore: Raffaele Loglisci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    f1_score
)
from tensorflow import keras
from tensorflow.keras import backend as K
from inspect import signature
from imblearn.over_sampling import SMOTE

K.clear_session()

# Caricamento del dataset
try:
    dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")

print(dataset.info())

# Selezione X e y, rimozione colonne inutili
y = dataset['Overall Survival Status'].str.upper().map({'0:LIVING': 0, '1:DECEASED': 1,
                                                        'ALIVE': 0, 'DEAD': 1})

# Rimuove identificativi e colonne prive di informazione
X = dataset.drop(["Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
                  "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
                  "Tumor Tissue Origin", "Last Communication Contact", "Oncotree Code", "Overall Survival Status"],
                 axis=1)

# Individua colonne non numeriche da trasformare (encoding)
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# IMPUTAZIONE dei valori mancanti prima di SMOTE
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Standardizzazione solo su colonne numeriche
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)


# Funzione: visualizza istogramma classi dopo SMOTE
def plot_class_balance(y, title="Distribuzione classi dopo SMOTE"):
    contatore = pd.Series(y).value_counts().sort_index()
    percentuali = contatore / contatore.sum() * 100
    df = pd.DataFrame({'Count': contatore, 'Percentuale': percentuali})
    print("\nTabella distribuzione classi dopo SMOTE:")
    print(df)
    df['Classe'] = df.index.astype(str)
    plt.figure(figsize=(5, 4))
    ax = sn.barplot(data=df, x='Classe', y='Percentuale', palette='Set2')
    for idx, row in df.iterrows():
        ax.text(idx, row['Percentuale'] + 1, f"{int(row['Count'])} ({row['Percentuale']:.1f}%)",
                ha='center', va='bottom', fontsize=10)
    plt.title(title)
    plt.ylim(0, 110)
    plt.ylabel('Percentuale (%)')
    plt.xlabel('Classe')
    plt.tight_layout()
    plt.show()


plot_class_balance(y_train_bal)

# DataFrame
X_train_bal_scaled = pd.DataFrame(X_train_bal_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# Definizione del modello
def create_model(neurons=64, activation='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_bal_scaled.shape[1],)),
        keras.layers.Dense(neurons, activation=activation),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = create_model()

model.fit(
    X_train_bal_scaled,
    y_train_bal,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test)
)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# CROSS-VALIDATION CON SMOTE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_train_bal_scaled):
    X_tr, X_val = X_train_bal_scaled.iloc[train_idx], X_train_bal_scaled.iloc[val_idx]
    y_tr, y_val = y_train_bal.iloc[train_idx], y_train_bal.iloc[val_idx]
    modelKF = create_model()
    modelKF.fit(X_tr, y_tr, epochs=30, batch_size=64, verbose=0)  # nessun class_weight qui
    val_loss, val_acc = modelKF.evaluate(X_val, y_val, verbose=0)
    cv_scores.append(val_acc)
    K.clear_session()

print('\ncv_scores mean:', np.mean(cv_scores))
print('cv_score variance:', np.var(cv_scores))
print('cv_score standard deviation:', np.std(cv_scores))

# Calcolo e visualizzazione AUC e ROC
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.round(y_pred_prob.flatten()).astype(int)

print('\nClassification report:\n', classification_report(y_test, y_pred))
print('\nConfusion matrix:\n', confusion_matrix(y_test, y_pred))

# Matrice di confusione normalizzata (percentuali)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Oranges')
plt.title('Matrice di confusione normalizzata (percentuali)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calcolo e visualizzazione AUC e ROC
probs = y_pred_prob.flatten()
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Precision-recall curve
average_precision = average_precision_score(y_test, probs)
precision, recall, _ = precision_recall_curve(y_test, probs)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='red', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='orange', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
plt.show()

# F1-score
f1 = f1_score(y_test, y_pred)
print('\nf1 score:', f1)

# Grafico per varianza/deviazione standard cv_scores
data = {'variance': np.var(cv_scores), 'standard deviation': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(names, values, color='orange')
plt.title('Varianza e deviazione standard dei cv_scores')
plt.show()
