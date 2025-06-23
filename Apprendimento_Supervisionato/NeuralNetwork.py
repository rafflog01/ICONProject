"""
@author: Raffaele Loglsci

"""
from inspect import signature
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, \
    average_precision_score, precision_recall_curve, f1_score
from tensorflow.keras import backend as K

K.clear_session()

# Caricamento del dataset
try:
    data = pd.read_csv("../2.Ontologia/breast-cancer.csv")
except FileNotFoundError:
    try:
        data = pd.read_csv("2.Ontologia/breast-cancer.csv")
    except FileNotFoundError:
        data = pd.read_csv("breast-cancer.csv")
data['diagnosis'] = data['diagnosis'].replace('M', 1)
data['diagnosis'] = data['diagnosis'].replace('B', 0)

# Esplorazione del dataset
print(data.info())

# Divido i dati in features di input e feature target
y = data['diagnosis']
X = data.drop(['id', 'diagnosis'], axis=1)
X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Divido il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converti X_train_scaled e X_test_scaled in DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)


# Costruisco il modello della rete neurale
def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Creazione istanza del modello e addestramento
model1 = create_model()
model1.fit(X_train_scaled, y_train, epochs=30, batch_size=64)

# Valutazione delle prestazioni del modello
test_loss, test_accuracy = model1.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Calcolo delle previsioni e arrotondamento
predictions = model1.predict(X_test_scaled)
rounded = [round(x[0]) for x in predictions]

# Stampa del report di classificazione e della matrice di confusione
print('\nClassification report:\n', classification_report(y_test, rounded))
print('\nConfusion matrix:\n', confusion_matrix(y_test, rounded))

# Creazione grafica della matrice di confusione con percentuali
conf_matrix = confusion_matrix(y_test, rounded)

# Normalizzazione della matrice di confusione (percentuali)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Creazione del DataFrame per la visualizzazione con percentuali
df_cm = pd.DataFrame(conf_matrix_percent, index=[i for i in "01"], columns=[i for i in "01"])

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='.2f', cmap='Oranges')

plt.title('Matrice di confusione normalizzata (percentuali)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# K-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# Esegui la cross-validation
for train_index, val_index in kf.split(X_train_scaled):
    X_train_fold, X_val_fold = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    modelK = create_model()
    modelK.fit(X_train_fold, y_train_fold, epochs=30, batch_size=64, verbose=0)

    val_loss, val_accuracy = modelK.evaluate(X_val_fold, y_val_fold)
    cv_scores.append(val_accuracy)
    K.clear_session()

# Stampa delle statistiche ottenute dalla cross-validation
print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score standard deviation:{}'.format(np.std(cv_scores)))

# Calcolo dell'AUC per la curva ROC
probs = model1.predict(X_test_scaled)[:, 0]
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

# Calcolo della curva ROC e visualizzazione
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FP RATE')
plt.ylabel('TP RATE')
plt.show()

# Calcolo della curva Precision-Recall e visualizzazione
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
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# Calcolo e visualizzazione dell'F1-score
f1 = f1_score(y_test, rounded)
print('\nf1 score: ', f1)

# Creazione di un grafico per visualizzare varianza e deviazione standard dei cv_scores
data = {'variance': np.var(cv_scores), 'standard deviation': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values, color='orange')
plt.show()
plt.show()