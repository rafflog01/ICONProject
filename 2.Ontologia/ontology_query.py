from owlready2 import *
import os

print("ONTOLOGIA\n")

# Ottieni il percorso assoluto del file
current_dir = os.path.dirname(os.path.abspath(__file__))
owl_file = os.path.join(current_dir, "breast_ontology.owl")

onto = get_ontology(owl_file).load()

# Stampa di tutte le classi dell'ontologia
print("####################################################################################################")
print("LISTA DELLE CLASSI DELL'ONTOLOGIA\n")
classes = list(onto.classes())
for cls in classes:
    print(f"• CLASSE: {cls.name}")
print()

# Stampa di tutte le object properties dell'ontologia
print("####################################################################################################")
print("LISTA DELLE OBJECT PROPERTIES DELL'ONTOLOGIA\n")
object_properties = list(onto.object_properties())
for prop in object_properties:
    print(f"• OBJECT PROPERTY: {prop.name}")
print()

# Stampa di tutte le data properties dell'ontologia
print("####################################################################################################")
print("LISTA DELLE DATA PROPERTIES DELL'ONTOLOGIA\n")
data_properties = list(onto.data_properties())
for prop in data_properties:
    print(f"• PROPERTY: {prop.name}")
print()

print("\n##################################################QUERIES##################################################")

# Query per ottenere pazienti con diagnosi "B" (Benigni)
diagnosis_B = onto.search_one(type=onto.Diagnosis, diagnosis="B")
query_result = list(onto.search(type=onto.Patient, hasDiagnosis=diagnosis_B))

print("\nPAZIENTI CON DIAGNOSI BENIGNA:")
for patient in query_result:
    print(f"• PAZIENTI: {patient.name}")

# Query per ottenere pazienti con diagnosi "M" (Maligni)
diagnosis_M = onto.search_one(type=onto.Diagnosis, diagnosis="M")
query_result = list(onto.search(type=onto.Patient, hasDiagnosis=diagnosis_M))

print("\nPAZIENTI CON DIAGNOSI MALIGNA:")
for patient in query_result:
    print(f"• PAZIENTI: {patient.name}")