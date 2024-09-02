import pandas as pd
from pyswip import Prolog

import pandas as pd

# Carica il dataset heart.csv
dataset = pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di Bari/Desktop/KNOWLEDGE BASE/diabetes.csv")

# Apri il file KB.pl in modalità scrittura
with open("KB_diabete.pl", "r") as prolog_file:
    prolog_file.write(":- discontiguous cp/1.")
    # Itera attraverso le righe del dataset
    for index, row in dataset.iterrows():
        age = row[0]  # Il primo valore
        tipo = row[2]  # Il terzo valore

        # Scrivi il fatto Prolog nel file KB_diabete.pl con age e tipo
        prolog_fact = f'age({age}).\n'
        prolog_file.write(prolog_fact)

        prolog_fact = f'cp({tipo}).\n'
        prolog_file.write(prolog_fact)


# Regole Prolog da scrivere nel file
prolog_regole = """
% Regola utile a verificare se la paziente è diabetica
paziente_diabetica(Gravidanze, Glucosio, PressioneSanguigna, SpessorePelle, Insulina, BMI, ProbabilitaDiabete, Eta, Diabetica) :-
    % Condizioni per essere diabetica
    (   
        (Gravidanze >= 0 ->  Cond1=1; Cond1=0),% Numero di gravidanze avute dalla paziente, non è un valore necessario al fine della valutazione del diabete
        (Glucosio >= 126 ->  Cond2=1; Cond2=0),% Livello di glucosio nel sangue 
        (PressioneSanguigna >= 80 -> Cond3=1; Cond3=0),% Valore della pressione sanguigna
        (SpessorePelle >= 20 ->  Cond4=1; Cond4=0),% Valore di spessore della pelle
        (Insulina >= 25 ->  Cond5=1; Cond5=0),% Livello di insulina
        (BMI >= 30 -> Cond6=1; Cond6=0), % Si consiglia un BMI più alto per indicare il rischio di diabete
        (ProbabilitaDiabete > 0.5 -> Cond7=1; Cond7=0), % Aumento della probabilità di diabete
        (Eta >= 20 -> Cond8=1; Cond8=0) % Età della paziente
    ),
    
    % Somma delle condizioni
    Somma is Cond1 + Cond2 + Cond3 + Cond4 + Cond5 + Cond6 + Cond7 + Cond8,

    % Se il numero di condizioni soddisfatte supera una soglia, la paziente è considerata diabetica
    (Somma >= 5 -> Diabetica = si; Diabetica = no).


% Regola per calcolare letà media delle pazienti che hanno il diabete
eta_media(EtàMedia) :-
(
    findall(Age, (age(Age), Age > 0), ListeEtà),% Estrazione delle età delle pazienti diabetiche
    length(ListeEtà, NumeroPersone),% Conta il numero di persone con diabete
    sum_list(ListeEtà, SommaEtà),% Somma delle età
    EtàMedia is SommaEtà/NumeroPersone.% Calcolo delletà media
).

"""

# Apri il file KB_diabete.pl in modalità append e scrivi le regole Prolog
with open("KB_diabete.pl", "a") as prolog_file:
    prolog_file.write(prolog_regole)


# Chiudi il file KB_diabete.pl
prolog_file.close()

# Crea un oggetto Prolog
prolog = Prolog()

# Carica il file KB.pl
prolog.consult("KB_diabete.pl")

# Esegui la query
print(list(prolog.query("eta_media(EtàMedia)")))