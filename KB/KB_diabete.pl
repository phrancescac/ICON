
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




% Regola per calcolare il rischio di diabete 
rischio_diabete(Glucosio, PressioneSanguigna, Insulina, BMI, ProbabilitaDiabete, Eta, Rischio) :-
    (   
    fattore_genetico(ProbabilitaDiabete, FattoreGenetico),
    fattore_glucosio(Glucosio, FattoreGlucosio),
    fattore_bmi(BMI, FattoreBMI),
    fattore_eta(Eta, FattoreEta),
    fattore_pressione(PressioneSanguigna, FattorePressione),
    fattore_insulina(Insulina, FattoreInsulina),
   
    % Somma ponderata dei vari fattori, con pesi che sommano a 1.0
    Rischio is (FattoreGenetico * 0.25) + (FattoreInsulina * 0.25) + 
                         (FattoreGlucosio * 0.2) + (FattoreBMI * 0.15) + 
                         (FattoreEta * 0.1) + (FattorePressione * 0.05)
	).

% Clausola per il fattore genetico 
fattore_genetico(ProbabilitaDiabete, FattoreGenetico) :-  
    (
        ProbabilitaDiabete > 1.0 -> FattoreGenetico = 1.0; % Alto rischio genetico
        ProbabilitaDiabete > 0.5 -> FattoreGenetico = 0.7; % Rischio moderato
        FattoreGenetico = 0.4 % Basso rischio genetico
    ).

% Clausola per il fattore glucosio
fattore_glucosio(Glucosio, 0.9) :- Glucosio >= 126.
fattore_glucosio(Glucosio, 0.5) :- Glucosio >= 100, Glucosio < 126.
fattore_glucosio(_, 0.1).

% Clausola per il fattore insulina
fattore_insulina(Insulina, 0.9 ) :- Insulina >= 25. % alto rischio
fattore_insulina(Insulina, 0.5) :- Insulina >=20, Insulina <25. % medio rischio
fattore_insulina(_, 0.2).

% Clausola per il fattore BMI
fattore_bmi(BMI, 0.9) :- BMI >= 30.
fattore_bmi(BMI, 0.5) :- BMI >= 25, BMI < 30.
fattore_bmi(_, 0.2).

% Clausola per il fattore età
fattore_eta(Eta, 0.9) :- Eta >= 60.
fattore_eta(Eta, 0.7) :- Eta >= 40, Eta < 60.
fattore_eta(Eta, 0.4) :- Eta >= 20, Eta < 40.
fattore_eta(_, 0.2).

% Clausola per il fattore pressione sanguigna
fattore_pressione(PressioneSanguigna, 0.7) :- PressioneSanguigna >= 80.
fattore_pressione(_, 0.2).


% Regola per verificare se la paziente è guarita
paziente_guarita(Glucosio, BMI, Insulina, Guarita) :-
    (   
    	glucosio_normale(Glucosio),
    	bmi_normale(BMI),
    	insulina_normale(Insulina),
    	Guarita = si
    ).

paziente_guarita(_, _, _, Guarita) :-
    Guarita = no.

% Clausola per determinare se il glucosio è normale
glucosio_normale(Glucosio) :- Glucosio < 100.

% Clausola per determinare se il BMI è normale
bmi_normale(BMI) :- BMI < 25.

% Clausola per determinare se linsulina è normale
insulina_normale(Insulina) :- Insulina < 30.



% Regola per identificare il tipo di diabete 
tipo_diabete(Insulina, Eta, BMI, PressioneSanguigna, Glucosio, tipo_1) :-
    Insulina < 25,                   % Insulina bassa
    Glucosio >= 126,                 % Livelli elevati di glucosio 
    Eta < 40,                        % Età sotto i 40 anni
    BMI < 30,                        % BMI generalmente inferiore (meno comune lobesità in Tipo 1)
    PressioneSanguigna < 80, !.      % Pressione sanguigna non alta (non associata a Tipo 1)

tipo_diabete(Insulina, Eta, BMI, PressioneSanguigna, Glucosio, tipo_2) :-
    Insulina >= 25,                  % Insulina normale o alta 
    Glucosio >= 126,                 % Livelli elevati di glucosio 
    Eta >= 40,                       % Età pari o superiore a 40 anni
    BMI >= 30,                       % Spesso associato a obesità o sovrappeso
    PressioneSanguigna >= 80, !.     % Pressione sanguigna alta o moderata

tipo_diabete(Insulina, Eta, BMI, PressioneSanguigna, Glucosio, prediabete) :-
    Glucosio >= 100,                 % Livelli di glucosio elevati ma non diabetici
    Glucosio < 126,
    BMI >= 25,                       % Sovrappeso ma non necessariamente obesità
    PressioneSanguigna >= 80, !.     % Pressione sanguigna moderata

tipo_diabete(Insulina, Eta, BMI, PressioneSanguigna, Glucosio, indeterminato) :-
    Insulina < 25,                   % Insulina bassa
    Glucosio >= 126,                 % Livelli elevati di glucosio 
    Eta >= 40, !.                    % Età avanzata ma bassa insulina: caso meno comune, indeterminato
