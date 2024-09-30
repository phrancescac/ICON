import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Caricamento del dataset
df = pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di Bari/Desktop/ICON-DIABETE/Dataset/diabetes.csv")

# Eliminazione dei duplicati
df = df.drop_duplicates()

# Gestione dei valori nulli nel dataset
for col in ["Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI"]:
    mean = df[df[col] != 0][col].mean()
    std = df[df[col] != 0][col].std()

    values = df[col].values
    np.random.seed(23)

    for i, val in enumerate(values):
        if val == 0:
            values[i] = mean + std * (np.random.rand() * 2 - 1) * 1.2
    df[col] = pd.Series(values).astype(df[col].dtype)

# Definizione delle variabili indipendenti (X) e target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Divisione del dataset in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

# Definizione dei classificatori da confrontare
classifiers = {
    'KNN': KNeighborsClassifier(),
    'DECISION TREE CLASSIFIER': DecisionTreeClassifier(),
    'RANDOM FOREST CLASSIFIER': RandomForestClassifier(),
    'ADABOOST CLASSIFIER': AdaBoostClassifier(),
    'XGBOOST CLASSIFIER': XGBClassifier()
}

# Dizionari per salvare le metriche di valutazione
accuracy_test = {}
cv_means = {}
cv_stds = {}

# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5)

# Calcolo delle metriche per ogni classificatore
for name, classifier in classifiers.items():
    # Cross-Validation per ogni classificatore (media e std)
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=kf, scoring='accuracy')
    cv_means[name] = cv_scores.mean()
    cv_stds[name] = cv_scores.std()

    # Addestramento del modello e valutazione sul test set
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy_test[name] = accuracy_score(y_test, y_pred)

# Creazione del grafico per la media delle accuracy della cross-validation e dei test set
plt.figure(figsize=(12, 8))

# Grafico a barre per le accuracy sul test set
plt.bar(accuracy_test.keys(), accuracy_test.values(), color='purple', label='Test Set Accuracy')

# Aggiunta della media e della deviazione standard della cross-validation
for i, name in enumerate(accuracy_test.keys()):
    plt.errorbar(i, cv_means[name], yerr=cv_stds[name], fmt='o', color='green', label='CV Mean ± Std' if i == 0 else "")

# Impostazioni del grafico
plt.title('Accuracy Scores of Different Classifiers (Test Set and Cross-Validation)', fontweight='bold')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Stampa dei risultati dettagliati
for name in classifiers.keys():
    print(f"Classifier: {name}")
    print(f"  Test Set Accuracy: {accuracy_test[name]:.4f}")
    print(f"  CV Mean Accuracy: {cv_means[name]:.4f}")
    print(f"  CV Std Dev: {cv_stds[name]:.4f}")
    print("-" * 40)
