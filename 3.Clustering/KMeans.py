import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

# Caricamento del dataset
try:
    dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")

dataset['Overall Survival Status'] = dataset['Overall Survival Status'].map(
    {'0:LIVING': 0, '1:DECEASED': 1, 'ALIVE': 0, 'DEAD': 1})
dataset.drop(columns=dataset.columns[dataset.columns.str.contains('unnamed', case=False)], inplace=True)

if 'Overall Primary Tumor Grade' in dataset.columns:
    dataset['Overall Primary Tumor Grade'] = dataset['Overall Primary Tumor Grade'].map({
        'I  Low Grade (Well Differentiated)': 1,
        'II  Intermediate Grade (Moderately Differentiated)': 2,
        'III High Grade (Poorly Differentiated)': 3,
        'Unknown': np.nan
    })
    mediana_grade = dataset['Overall Primary Tumor Grade'].median(skipna=True)
    dataset['Overall Primary Tumor Grade'] = dataset['Overall Primary Tumor Grade'].fillna(mediana_grade)

# Selezione delle colonne rilevanti
relevant_cols = [
    'ER Status of Sequenced Sample',
    'PR Status of Sequenced Sample',
    'HER2 IHC Status of Sequenced Sample',
    'Invasive Carcinoma Diagnosis Age',
    'Mutation Count',
    'Overall Survival Status',
    'Overall Patient HR Status',
    'Overall Primary Tumor Grade',
    'Sex',
    'Stage At Diagnosis',
    'TMB (nonsynonymous)',
    'Disease Free Event',
    'Primary Tumor Laterality',
    'Overall Survival (Months)',
]
dataset = dataset[relevant_cols]

# Sostituzione Positive/Negative/Equivocal/Unk/ND con valori numerici
for col in [
    'ER Status of Sequenced Sample',
    'PR Status of Sequenced Sample',
    'HER2 IHC Status of Sequenced Sample',
    'Overall HR Status of Sequenced Sample',
    'Overall HER2 Status of Sequenced Sample',
    'Overall Patient HR Status'
]:
    if col in dataset.columns:
        dataset[col] = dataset[col].map({
            'Positive': 1,
            'Negative': 0,
            'Equivocal': 0.5,
            'Unk/ND': np.nan,
            'Left': 1,
            'Right': 0,
        })

if 'Sex' in dataset.columns:
    dataset['Sex'] = dataset['Sex'].map({'Female': 0, 'Male': 1})

# Gestione missing values
dataset.fillna(dataset.mode().iloc[0], inplace=True)

# Prepara X per il clustering (solo le colonne rilevanti già selezionate)
X = dataset.copy()

# Individua colonne non numeriche
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Individua tutte le colonne categoriche attualmente presenti in X
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Gestione di eventuali NaN dopo One-Hot Encoding
X.fillna(0, inplace=True)

# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calcolo WCSS e silhouette score
wcss = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
    else:
        silhouette_scores.append(0)

# Elbow plot
plt.plot(k_range, wcss, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Addestramento finale
kmeans_final = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans_final.fit(X_pca)

print(f"\n-WCSS: {kmeans_final.inertia_:.2f}")
print(f"-Silhouette Score: {silhouette_score(X_pca, kmeans_final.labels_):.4f}")

# Assegna etichetta cluster al dataset originale (già costruito come X)
dataset['cluster'] = kmeans_final.labels_

# Riordina eventualmente le colonne
columns_order = list(dataset.columns)
if 'diagnosis' in columns_order:
    columns_order.remove('diagnosis')
    columns_order.append('diagnosis')
if 'cluster' in columns_order:
    columns_order.remove('cluster')
    columns_order.insert(-1, 'cluster')

dataset_reordered = dataset[columns_order]
dataset_reordered.to_csv('breast_msk_2018_clinical_data-clusters.csv', index=False)

# Incrocia cluster e classe reale
confusion = pd.crosstab(dataset['cluster'], dataset['Overall Survival Status'])

print('\nTabella incrocio cluster vs classe reale:')
print(confusion)

# Visualizzazione cluster rispetto alla label reale su PCA
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=dataset['Overall Survival Status'],
                style=kmeans_final.labels_,
                palette='coolwarm')

plt.title("Cluster KMeans rispetto allo status di sopravvivenza reale")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="Sopravvivenza / Cluster")
plt.show()

outlier = dataset[kmeans_final.labels_ == 2]
print(outlier)
ari = adjusted_rand_score(dataset['Overall Survival Status'], dataset['cluster'])
print('Adjusted Rand Index:', ari)  # Più è basso più cluster e classi "non coincidono"

# Statistiche descrittive per ogni cluster
print(dataset.groupby('cluster').mean(numeric_only=True))
