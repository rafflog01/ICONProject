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
from tensorflow.keras.callbacks import EarlyStopping
from inspect import signature
from imblearn.over_sampling import SMOTE

K.clear_session()

# Caricamento dataset

try:
    dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")

print(dataset.info())

# Selezione X e y
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


#  Definizione del modello Keras
def create_model(neurons=64, activation='relu', lr=1e-3, input_dim=None, seed=42):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(neurons, activation=activation),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Inner 5-fold CV per tuning (manuale, senza leakage)
def cv_eval_config(cfg, X_train_raw, y_train, seed=42, n_splits=5):
    """
    Valuta una configurazione iperparametri con 5-fold CV sul SOLO training:
    - Imputer/SMOTE/Scaler sono fittati solo su train della fold
    - Ritorna la media di val_accuracy sulle fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_accs = []

    for tr_idx, val_idx in skf.split(X_train_raw, y_train):
        X_tr_raw, X_val_raw = X_train_raw.iloc[tr_idx], X_train_raw.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Imputazione
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_raw)
        X_val_imp = imputer.transform(X_val_raw)

        # SMOTE SOLO sul train della fold
        sm = SMOTE(random_state=seed)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_imp, y_tr)

        # Scaling
        scaler = StandardScaler()
        X_tr_bal_scaled = scaler.fit_transform(X_tr_bal)
        X_val_scaled = scaler.transform(X_val_imp)

        # Modello
        K.clear_session()
        model = create_model(
            neurons=cfg["neurons"],
            activation=cfg["activation"],
            lr=cfg["lr"],
            input_dim=X_tr_bal_scaled.shape[1],
            seed=seed
        )
        es = EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True, verbose=0)
        history = model.fit(
            X_tr_bal_scaled, y_tr_bal,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_data=(X_val_scaled, y_val),
            callbacks=[es],
            verbose=0
        )
        # Migliore val_accuracy nella fold
        fold_val_acc = float(np.max(history.history.get("val_accuracy", [np.nan])))
        val_accs.append(fold_val_acc)

        K.clear_session()

    return float(np.nanmean(val_accs))


# Repeated Holdout (outer) + Inner 5-fold CV (tuning)
n_runs = 10
test_size = 0.20
seeds = list(range(42, 42 + n_runs))
plot_each_run = True

acc_list, f1_list, auc_list, ap_list = [], [], [], []

# Griglia iperparametri per la rete
param_grid = [
    {"neurons": 32, "activation": "relu", "lr": 1e-3, "batch_size": 32, "epochs": 40},
    {"neurons": 64, "activation": "relu", "lr": 1e-3, "batch_size": 32, "epochs": 40},
    {"neurons": 64, "activation": "relu", "lr": 5e-4, "batch_size": 64, "epochs": 50},
    {"neurons": 64, "activation": "relu", "lr": 5e-4, "batch_size": 64, "epochs": 50},
    {"neurons": 128, "activation": "relu", "lr": 1e-3, "batch_size": 64, "epochs": 50},
    {"neurons": 64, "activation": "tanh", "lr": 1e-3, "batch_size": 32, "epochs": 40},
]

for i, seed in enumerate(seeds, start=1):
    # Outer split 80/20 (stratificato)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y
    )

    # Inner 5-fold CV: selezione migliore configurazione su SOLO X_train
    best_cfg, best_cv_acc = None, -np.inf
    for cfg in param_grid:
        cv_acc = cv_eval_config(cfg, X_train, y_train, seed=seed, n_splits=5)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_cfg = cfg

    print(f"[Run {i}/{n_runs} - seed={seed}] Inner 5-fold best ValAcc={best_cv_acc:.4f} | cfg={best_cfg}")

    # Refit finale su TUTTO il training dell'outer split (no leakage)
    # Imputazione sul training intero
    imputer_final = SimpleImputer(strategy="median")
    X_train_imp = imputer_final.fit_transform(X_train)
    X_test_imp = imputer_final.transform(X_test)

    # SMOTE solo sul training
    sm_final = SMOTE(random_state=seed)
    X_train_bal, y_train_bal = sm_final.fit_resample(X_train_imp, y_train)

    # Scaling (fit su train, transform su test)
    scaler_final = StandardScaler()
    X_train_bal_scaled = scaler_final.fit_transform(X_train_bal)
    X_test_scaled = scaler_final.transform(X_test_imp)

    # Modello finale con best_cfg
    K.clear_session()
    best_model = create_model(
        neurons=best_cfg["neurons"],
        activation=best_cfg["activation"],
        lr=best_cfg["lr"],
        input_dim=X_train_bal_scaled.shape[1],
        seed=seed
    )
    es_final = EarlyStopping(monitor="accuracy", patience=5, mode="max", restore_best_weights=True, verbose=0)
    best_model.fit(
        X_train_bal_scaled, y_train_bal,
        epochs=best_cfg["epochs"],
        batch_size=best_cfg["batch_size"],
        callbacks=[es_final],
        verbose=0
    )

    # Valutazione sul test dell'outer split
    y_pred_prob = best_model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1v = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        auc = np.nan
    ap = average_precision_score(y_test, y_pred_prob)

    acc_list.append(acc)
    f1_list.append(f1v)
    auc_list.append(auc)
    ap_list.append(ap)

    print(f"    Test: Acc={acc:.4f} F1={f1v:.4f} AUC={auc:.4f} AP={ap:.4f}")
    print("\nClassification report (test):\n", classification_report(y_test, y_pred))

    # Grafici per questo run
    if plot_each_run:
        # Matrice di confusione normalizzata
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Oranges')
        plt.title(f'Matrice di confusione normalizzata (%) - Test (Run {i})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

        # ROC Curve
        if not np.isnan(auc):
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            plt.figure(figsize=(6, 5))
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Test (Run {i}) | AUC={auc:.3f}')
            plt.tight_layout()
            plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.figure(figsize=(6, 5))
        plt.step(recall, precision, color='red', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='orange', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall (AP={ap:.3f}) - Test (Run {i})')
        plt.tight_layout()
        plt.show()

    # Pulisce la sessione per evitare memory leak
    K.clear_session()


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
    ax.boxplot(vals, vert=True, patch_artist=True, boxprops=dict(facecolor='#FF9F80'))
    ax.set_title(f'{name} – distribuzione su {n_runs} run')
    ax.set_ylabel(name)

plt.tight_layout()
plt.show()

# Grafico varianza e deviazione standard (accuracy)
fig, ax = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
ax.bar(['variance', 'std dev'], [acc_var, acc_std], color=['#F17CB0', '#60BD68'])
for i, v in enumerate([acc_var, acc_std]):
    ax.text(i, v + max(1e-3, v) * 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
ax.set_title('Stabilità (Accuracy) – varianza e std su test')
plt.tight_layout()
plt.show()
