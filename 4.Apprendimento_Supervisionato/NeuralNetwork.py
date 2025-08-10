"""
@autore: Raffaele Loglisci
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)
from tensorflow import keras
from tensorflow.keras import backend as K
from inspect import signature
from imblearn.over_sampling import SMOTE

K.clear_session()

# ===============================
# 1) Caricamento del dataset
# ===============================
try:
    dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")

print(dataset.info())

# ===============================
# 2) Selezione X e y
# ===============================
y = dataset['Overall Survival Status'].str.upper().map({
    '0:LIVING': 0, '1:DECEASED': 1, 'ALIVE': 0, 'DEAD': 1
})

# Rimuove identificativi e colonne prive di informazione
X = dataset.drop([
    "Study ID", "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Site of Sample", "Sample Type", "Sex", "Tumor Sample Histology",
    "Tumor Tissue Origin", "Last Communication Contact", "Oncotree Code", "Overall Survival Status"
], axis=1)

# One-Hot Encoding delle categoriche
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ===============================
# 3) Split train/test (holdout)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# ===============================
# 4) Imputazione + SMOTE + Scaling sul train (no leakage)
# ===============================
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imp, y_train)

scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test_imp)

# Per comodità, come DataFrame (solo per eventuale ispezione)
X_train_bal_scaled = pd.DataFrame(X_train_bal_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# ===============================
# 5) Funzione di utilità: distribuzione classi
# ===============================
def plot_class_balance(y_series, title="Distribuzione classi dopo SMOTE"):
    contatore = pd.Series(y_series).value_counts().sort_index()
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

# ===============================
# 6) Definizione del modello Keras
# ===============================
def create_model(neurons=64, activation='relu', optimizer='adam', input_dim=None):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(neurons, activation=activation),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Istanziamento per training su holdout
model = create_model(input_dim=X_train_bal_scaled.shape[1])

# ===============================
# 7) Training su holdout e valutazione
# ===============================
model.fit(
    X_train_bal_scaled,
    y_train_bal,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'\n[Test] Accuracy: {test_acc:.4f}')

# Predizioni sul test
y_pred_prob = model.predict(X_test_scaled, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print('\nClassification report (test):\n', classification_report(y_test, y_pred))
print('\nConfusion matrix (test):\n', confusion_matrix(y_test, y_pred))

# Matrice di confusione normalizzata
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Oranges')
plt.title('Matrice di confusione normalizzata (percentuali) - Test')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC e PR su test
try:
    auc_test = roc_auc_score(y_test, y_pred_prob)
    print(f'AUC (test): {auc_test:.3f}')
except ValueError:
    print('AUC (test): non calcolabile (una sola classe presente)')

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test')
plt.show()

ap_test = average_precision_score(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='red', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='orange', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall curve (AP={ap_test:.2f}) - Test')
plt.show()

print(f'F1 (test): {f1_score(y_test, y_pred):.4f}')

# ===============================
# 8) 5-fold CV senza leakage con metriche per fold
# ===============================
print("\n=== 5-fold CV senza leakage: metriche per fold ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_folds, f1_folds, roc_folds, ap_folds = [], [], [], []
fold_idx = 1

for tr_idx, val_idx in cv.split(X, y):
    # Split grezzo (dopo OHE già effettuato su X completo)
    X_tr_raw, X_val_raw = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # Imputazione (fit sul train, transform su val)
    imputer_cv = SimpleImputer(strategy="median")
    X_tr_imp = imputer_cv.fit_transform(X_tr_raw)
    X_val_imp = imputer_cv.transform(X_val_raw)

    # SMOTE SOLO sul train (no leakage)
    smote_cv = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = smote_cv.fit_resample(X_tr_imp, y_tr)

    # Scaling (fit sul train bilanciato, applica a val)
    scaler_cv = StandardScaler()
    X_tr_bal_scaled = scaler_cv.fit_transform(X_tr_bal)
    X_val_scaled = scaler_cv.transform(X_val_imp)

    # Modello per fold
    model_cv = create_model(input_dim=X_tr_bal_scaled.shape[1])

    # Addestramento fold
    model_cv.fit(
        X_tr_bal_scaled, y_tr_bal,
        epochs=30,
        batch_size=64,
        verbose=0
    )

    # Predizioni su validation della fold
    probs = model_cv.predict(X_val_scaled, verbose=0).ravel()
    y_pred_fold = (probs >= 0.5).astype(int)

    # Metriche per fold
    acc = accuracy_score(y_val, y_pred_fold)
    f1v = f1_score(y_val, y_pred_fold)
    try:
        roc = roc_auc_score(y_val, probs)
    except ValueError:
        roc = np.nan  # nel raro caso di una sola classe in val
    ap = average_precision_score(y_val, probs)

    acc_folds.append(acc)
    f1_folds.append(f1v)
    roc_folds.append(roc)
    ap_folds.append(ap)

    print(f"[Fold {fold_idx}] Acc={acc:.4f} | F1={f1v:.4f} | ROC-AUC={roc:.4f} | AP={ap:.4f}")
    K.clear_session()
    fold_idx += 1

# Riepilogo CV
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

vals = [np.nanvar(acc_arr), acc_std]
labels = ['variance', 'standard dev']

fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(labels, vals, color=['#5DA5DA', '#60BD68'])
for i, v in enumerate(vals):
    ax.text(i, v + max(1e-3, v) * 0.05, f"{v:.4f}", ha='center', va='bottom', fontsize=9)
ax.set_title('Stabilità CV (Accuracy)')
plt.tight_layout()
plt.show()
