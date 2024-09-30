import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

# Parametri KNN
k_folds = 5
n_runs = 10
n_neighbors_values = range(1, 30)  # Valori di n_neighbors da testare

# Creazione delle liste per salvare i risultati
cv_means = []
cv_stds = []
test_accuracies = []

# Stratified KFold per mantenere la distribuzione delle classi
kf = StratifiedKFold(n_splits=k_folds)

# Liste per accuracies di train e test per ogni n_neighbors
accuracy_train_knn = []
accuracy_test_knn = []

for run in range(1, n_runs + 1):
    # Suddivisione in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=run)
    
    # Ciclo su ogni valore di n_neighbors
    for neighbors in n_neighbors_values:
        # Creo il classificatore KNN
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        
        # Addestramento sul training set
        knn.fit(X_train, y_train)
        
        # Calcolo delle accuracy per train e test
        accuracy_train_knn.append(knn.score(X_train, y_train))
        accuracy_test_knn.append(knn.score(X_test, y_test))
    
    # Cross-validation su training set per ogni run (con un valore fisso di n_neighbors, es. 5)
    knn_cv = KNeighborsClassifier(n_neighbors=5)
    cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Salvataggio dei risultati di cross-validation
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())
    
    # Predizione sul test set con n_neighbors = 5
    knn_cv.fit(X_train, y_train)
    y_pred = knn_cv.predict(X_test)
    test_accuracies.append(accuracy_score(y_test, y_pred))

# Visualizzazione dei risultati delle cross-validation
for run in range(n_runs):
    print(f"Run {run + 1}:")
    print(f"  Cross-validation accuracy mean: {cv_means[run]:.4f}")
    print(f"  Cross-validation accuracy std: {cv_stds[run]:.4f}")
    print(f"  Test set accuracy: {test_accuracies[run]:.4f}\n")

# Creazione del grafico per n_neighbors
plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_values, accuracy_train_knn[:len(n_neighbors_values)], label='Train', color='purple')
plt.plot(n_neighbors_values, accuracy_test_knn[:len(n_neighbors_values)], label='Test', color='green')

# Impostazione del titolo e delle etichette degli assi
plt.title('KNN - Number of Neighbors')
plt.xlabel('Number of Neighbors (n_neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 30))
plt.show()

# Creazione dei grafici
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_runs+1), test_accuracies, marker='o', color='purple', label='Test Accuracy')
plt.title('Test Set Accuracy for Each Run of KNN')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot delle accuratezze medie della cross-validation
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, n_runs+1), cv_means, yerr=cv_stds, fmt='-o', color='green', label='CV Mean Accuracy')
plt.title('Cross-Validation Accuracy (Mean and Std) for Each Run of KNN')
plt.xlabel('Run')
plt.ylabel('CV Accuracy (Mean ± Std)')
plt.legend()
plt.grid(True)
plt.show()
