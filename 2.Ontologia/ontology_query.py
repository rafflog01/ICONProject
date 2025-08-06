"""
@autore: Raffaele Loglisci
"""

from owlready2 import *
import os


def main():
    print("=== CARICAMENTO ONTOLOGIA ===\n")

    # Percorso file ontologia
    current_dir = os.path.dirname(os.path.abspath(__file__))
    owl_file = os.path.join(current_dir, "breast_ontology.owl")

    # Caricamento ontologia
    onto = get_ontology(owl_file).load()

    # Stampa Classi
    print("\n########################################")
    print("LISTA CLASSI DELL'ONTOLOGIA\n")
    for cls in onto.classes():
        print(f"• CLASSE: {cls.name}")

    # Stampa Object Properties
    print("\n########################################")
    print("LISTA OBJECT PROPERTIES DELL'ONTOLOGIA\n")
    for prop in onto.object_properties():
        print(f"• OBJECT PROPERTY: {prop.name}")

    # Stampa Data Properties
    print("\n########################################")
    print("LISTA DATA PROPERTIES DELL'ONTOLOGIA\n")
    for prop in onto.data_properties():
        print(f"• DATA PROPERTY: {prop.name}")

    print("\n########## QUERY ESEMPI ##########")

    # Esempio: Pazienti con Alta Probabilità di Sopravvivenza
    AltaProb = getattr(onto, "AltaProbabilitaSopravvivenza", None)
    if AltaProb:
        risultati = onto.search(type=AltaProb)
        print("\nPazienti con ALTA PROBABILITA' DI SOPRAVVIVENZA:")
        for r in risultati:
            print(f"• PAZIENTE: {r.name}")
        if not risultati:
            print("Nessun paziente trovato.")
    else:
        print("\nClasse 'AltaProbabilitaSopravvivenza' non presente nell'ontologia.")

    # Esempio: Pazienti con Bassa Probabilità di Sopravvivenza
    BassaProb = getattr(onto, "BassaProbabilitaSopravvivenza", None)
    if AltaProb:
        risultati = onto.search(type=BassaProb)
        print("\nPazienti con BASSA PROBABILITA' DI SOPRAVVIVENZA:")
        for r in risultati:
            print(f"• PAZIENTE: {r.name}")
        if not risultati:
            print("Nessun paziente trovato.")
    else:
        print("\nClasse 'BassaProbabilitaSopravvivenza' non presente nell'ontologia.")

    # Esempio: Pazienti ad Alto Rischio di Recidiva
    AltoRischio = getattr(onto, "PazienteAltoRischioRecidiva", None)
    if AltoRischio:
        risultati = onto.search(type=AltoRischio)
        print("\nPazienti ad ALTO RISCHIO DI RECIDIVA:")
        for r in risultati:
            print(f"• PAZIENTE: {r.name}")
        if not risultati:
            print("Nessun paziente trovato.")
    else:
        print("\nClasse 'PazienteAltoRischioRecidiva' non presente nell'ontologia.")

    # Esempio: Tumori presenti e i relativi pazienti (se relazioni definite)
    Tumore = getattr(onto, "Tumore", None)
    if Tumore:
        tumori = list(onto.search(type=Tumore))
        print(f"\nNUMERO DI TUMORI PRESENTI: {len(tumori)}")
        for t in tumori:
            print(f"• TUMORE: {t.name}")
        if not tumori:
            print("Nessun tumore trovato.")
    else:
        print("\nClasse 'Tumore' non presente nell'ontologia.")


if __name__ == "__main__":
    main()
