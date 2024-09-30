import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Percorso del dataset
file_path = "C:/Users/fracu/OneDrive - Università degli Studi di Bari/Desktop/ICON-DIABETE/Dataset/diabetes.csv"

# Caricamento del dataset
df = pd.read_csv(file_path)

# Rimozione dei duplicati
df = df.drop_duplicates()

# Gestione dei valori nulli nel dataset
columns_to_check = ["Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI"]
for col in columns_to_check:
    if col in df.columns:
        mean = df[df[col] != 0][col].mean()
        std = df[df[col] != 0][col].std()

        values = df[col].values
        np.random.seed(23)

        for i, val in enumerate(values):
            if val == 0:
                values[i] = mean + std * (np.random.rand() * 2 - 1) * 1.2
        df[col] = pd.Series(values).astype(df[col].dtype)

# Definizione delle features e del target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Parametri AdaBoost
n_estimators = 500
k_folds = 5
n_runs = 10
learning_rate = 0.05

# Creazione delle liste per salvare i risultati
cv_means = []
cv_stds = []
test_accuracies = []

# Stratified KFold per mantenere la distribuzione delle classi
kf = StratifiedKFold(n_splits=k_folds)

for run in range(1, n_runs + 1):
    # Suddivisione in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=run)
    
    # Modello AdaBoost con DecisionTree come base estimator
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                                  n_estimators=n_estimators, learning_rate=learning_rate, random_state=run)

    # Cross-validation su training set per ogni run
    cv_scores = cross_val_score(adaboost, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Addestramento del modello e predizione
    adaboost.fit(X_train, y_train)
    y_pred = adaboost.predict(X_test)

    # Salvataggio dei risultati
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())
    test_accuracies.append(accuracy_score(y_test, y_pred))

# Visualizzazione dei risultati
for run in range(n_runs):
    print(f"Run {run + 1}:")
    print(f"  Cross-validation accuracy mean: {cv_means[run]:.4f}")
    print(f"  Cross-validation accuracy std: {cv_stds[run]:.4f}")
    print(f"  Test set accuracy: {test_accuracies[run]:.4f}\n")

# Creazione dei grafici (senza matrice di confusione)
# Grafico delle accuratezze dei valori di test per ogni run
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_runs+1), test_accuracies, marker='o', color='purple', label='Test Accuracy')
plt.title('Test Set Accuracy for Each Run of AdaBoost')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Grafico delle accuratezze medie della cross-validation
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, n_runs+1), cv_means, yerr=cv_stds, fmt='-o', color='green', label='CV Mean Accuracy')
plt.title('Cross-Validation Accuracy (Mean and Std) for Each Run of AdaBoost')
plt.xlabel('Run')
plt.ylabel('CV Accuracy (Mean ± Std)')
plt.legend()
plt.grid(True)
plt.show()

# Parametri per analizzare diverse combinazioni di n_estimators e learning_rate
n_values_estimators = range(100, 1001, 100)
learning_rates = np.arange(0.1, 1.1, 0.1)

# Creazione di un subplot per visualizzare grafici con più valori di accuratezza
fig, axes = plt.subplots(5, 2, figsize=(15, 30))

for i, n_estimators in enumerate(n_values_estimators):
    train_accuracy = []
    test_accuracy = []

    for learning_rate in learning_rates:
        ada = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

        ada.fit(X_train, y_train)
        train_pred = ada.predict(X_train)
        test_pred = ada.predict(X_test)

        train_accuracy.append(accuracy_score(y_train, train_pred))
        test_accuracy.append(accuracy_score(y_test, test_pred))
        
    # Axes
    ax = axes[i//2, i%2]

    ax.plot(learning_rates, train_accuracy, label=f'Train Accuracy, n_estimators={n_estimators}', color='purple')
    ax.plot(learning_rates, test_accuracy, label=f'Test Accuracy, n_estimators={n_estimators}', color='green')

    ax.set_title(f'n_estimators={n_estimators}', fontweight='bold')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(np.arange(0.1,1.01,0.1))

plt.tight_layout()
plt.show()
