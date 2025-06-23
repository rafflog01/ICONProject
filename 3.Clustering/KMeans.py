"""
@author: Raffaele Loglsci

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Caricamento del dataset
try:
    dataset = pd.read_csv("breast-cancer.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast-cancer.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("2.Ontologia/breast-cancer.csv")
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
dataset.drop(columns=dataset.columns[dataset.columns.str.contains('unnamed', case=False)], inplace=True)

# Rimuovere la colonna 'id' e 'diagnosis' per il clustering
X = dataset.drop(['id', 'diagnosis'], axis=1)

# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcolo WCSS per diversi valori di K
wcss = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    # Il silhouette score non può essere calcolato per k=1
    if k > 1:
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    else:
        silhouette_scores.append(0)  # Per k=1 impostiamo il silhouette score a 0

# Elbow plot
plt.plot(k_range, wcss, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')

plt.show()

# Addestramento del modello finale con K=2
kmeans_final = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_final.fit(X_scaled)

# Stampa metriche
print(f"\n-WCSS: {kmeans_final.inertia_:.2f}")
print(f"-Silhouette Score: {silhouette_score(X_scaled, kmeans_final.labels_):.4f}")

# Aggiunta etichetta cluster al dataset originale
dataset['cluster'] = kmeans_final.labels_

# Riordina le colonne del DataFrame
columns_order = list(dataset.columns)
columns_order.remove('diagnosis')
columns_order.append('diagnosis')
columns_order.remove('cluster')
columns_order.insert(-1, 'cluster')
dataset_reordered = dataset[columns_order]

dataset_reordered.to_csv('breast-cancer_clusters.csv', index=False)
