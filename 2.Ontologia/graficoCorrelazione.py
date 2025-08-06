"""
@autore: Raffaele Loglisci
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento dati
dataset = None
try:
    dataset = pd.read_csv("breast_msk_2018_clinical_data.csv")
except FileNotFoundError:
    try:
        dataset = pd.read_csv("../2.Ontologia/breast_msk_2018_clinical_data.csv")
    except FileNotFoundError:
        try:
            dataset = pd.read_csv("2.Ontologia/breast_msk_2018_clinical_data.csv")
        except FileNotFoundError:
            print("ERRORE: File breast_msk_2018_clinical_data non trovato in nessuno dei percorsi.")
            exit(1)

if 'Somatic Status' in dataset.columns:
    dataset = dataset.drop('Somatic Status', axis=1)

# Pulizia nomi colonne: rimuove eventuali spazi finali
dataset.columns = dataset.columns.str.strip()

# INIZIO MAPPATURE ESTESE
dataset['Overall Survival Status'] = dataset['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})

for col in ['Overall HER2 Status of Sequenced Sample', 'Overall Patient HER2 Status']:
    if col in dataset.columns:
        dataset[col] = dataset[col].map({
            'Negative': 0, 'Positive': 1, 'Equivocal': -1,
            'Unk/ND': -2, 'Unknown': -2, 'HR+/HER2_Unknown': -2,
            'HR+/HER2-': 2, 'HR+/HER2+': 3, 'Triple Negative': 4,
            'HR+/HER2_Equivocal': -1
        })

for col in ['Overall Patient HR Status', 'Overall Patient Receptor Status', 'Receptor Status Primary']:
    if col in dataset.columns:
        dataset[col] = dataset[col].map({
            'Negative': 0, 'Positive': 1, 'HR+/HER2-': 2,
            'HR+/HER2+': 3, 'Triple Negative': 4
        })

# Mapping grado istologico
if 'Overall Primary Tumor Grade' in dataset.columns:
    dataset['Overall Primary Tumor Grade'] = dataset['Overall Primary Tumor Grade'].map({
        'I  Low Grade (Well Differentiated)': 1,
        'II  Intermediate Grade (Moderately Differentiated)': 2,
        'III High Grade (Poorly Differentiated)': 3,
        'Unknown': np.nan
    })
    grade_median = dataset['Overall Primary Tumor Grade'].median(skipna=True)
    dataset['Overall Primary Tumor Grade'] = dataset['Overall Primary Tumor Grade'].fillna(grade_median)

for col in ['PR Status of Sequenced Sample', 'PR Status of the Primary']:
    if col in dataset.columns:
        dataset[col] = dataset[col].map({'Negative': 0, 'Positive': 1})

# Funzioni per estrazione numerica da Stage
import re


def extract_numeric_stage(stage):
    if pd.isnull(stage):
        return np.nan
    m = re.match(r'([IV]+)', str(stage))
    roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    return roman[m.group(1)] if m else np.nan


if 'Stage At Diagnosis' in dataset.columns:
    dataset['Stage At Diagnosis'] = dataset['Stage At Diagnosis'].apply(extract_numeric_stage)


def extract_numeric_tstage(tstage):
    if pd.isnull(tstage):
        return np.nan
    m = re.match(r'T(\d+)', str(tstage))
    return int(m.group(1)) if m else np.nan


if 'T Stage' in dataset.columns:
    dataset['T Stage'] = dataset['T Stage'].apply(extract_numeric_tstage)

for col in ['Prior Breast Primary', 'Prior Local Recurrence']:
    if col in dataset.columns:
        dataset[col] = dataset[col].map({'No': 0, 'Yes': 1})

if 'Somatic Status' in dataset.columns:
    dataset['Somatic Status'] = dataset['Somatic Status'].map({'Matched': 1})

if 'Sex' in dataset.columns:
    dataset['Sex'] = dataset['Sex'].map({'Female': 0, 'Male': 1})

if "Patient's Vital Status" in dataset.columns:
    dataset["Patient's Vital Status"] = dataset["Patient's Vital Status"].map({'Alive': 0, 'Deceased': 1})

# Riempimento dei NaN
dataset = dataset.fillna(0)

# Seleziona tutte le colonne numeriche
numerical_columns = dataset.select_dtypes(include=['number']).columns.tolist()

# Calcolo e visualizzazione heatmap
corr_matrix = dataset[numerical_columns].corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm")
plt.title("Heat Map Correlazioni Breast Cancer (dato clinico + mapping esteso)")
plt.tight_layout()
plt.show()
