import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Caricamento dati
dataset = None
try:
    dataset = pd.read_csv("breast-cancer.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast-cancer.csv")
    except FileNotFoundError:
        try:
            dataset = pd.read_csv("2.Ontologia/breast-cancer.csv")
        except FileNotFoundError:
            print("ERRORE: File breast-cancer.csv non trovato in nessuno dei percorsi.")
            exit(1)

# Pulizia nomi colonne: rimuove eventuali spazi finali
dataset.columns = dataset.columns.str.strip()

# Elimina la colonna 'id' se presente (dato identificativo non utile per l'analisi)
if 'id' in dataset.columns:
    dataset = dataset.drop('id', axis=1)

# Codifica la label 'diagnosis': M = 1 (maligno), B = 0 (benigno)
if 'diagnosis' in dataset.columns:
    dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})

# Individua tutte le colonne numeriche per la heatmap
numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Genera matrice di correlazione
corr_matrix = dataset[numerical_columns].corr()

# Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm", cbar=True)
plt.title("Heat Map delle Correlazioni Breast Cancer (feature codificate)")
plt.tight_layout()
plt.show()