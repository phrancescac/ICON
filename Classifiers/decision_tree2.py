import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# Parametri Decision Tree
k_folds = 5
n_runs = 10
max_depth_values = range(1, 16)
min_samples_leaf_values = range(5, 31, 5)

# Creazione delle liste per salvare i risultati
cv_means = []
cv_stds = []

# Stratified KFold per mantenere la distribuzione delle classi
kf = StratifiedKFold(n_splits=k_folds)

# Liste per accuracies di train e test
accuracy_train = []
accuracy_test = []

for run in range(1, n_runs + 1):
    # Suddivisione in train e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=run)
    
    # Ciclo su ogni valore dell'albero del max_depth
    for depth in max_depth_values:
        # Creo il classificatore Decision Tree
        dtc = DecisionTreeClassifier(max_depth=depth, random_state=42)
        # Addestramento sul training set
        dtc.fit(X_train, y_train)
        # Calcolo delle accuracy per train e test
        accuracy_train.append(dtc.score(X_train, y_train))
        accuracy_test.append(dtc.score(X_test, y_test))
    
    # Creazione del grafico per max_depth
    plt.plot(max_depth_values, accuracy_train[:len(max_depth_values)], label='Train', color='purple')
    plt.plot(max_depth_values, accuracy_test[:len(max_depth_values)], label='Test', color='green')

    plt.title('MAX_DEPTH')
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, 16))
    plt.show()

    # Esplorazione del min_samples_leaf
    accuracy_train = []
    accuracy_test = []

    for leaf in min_samples_leaf_values:
        # Creo il classificatore Decision Tree con min_samples_leaf
        dtc = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        # Addestramento sul training set
        dtc.fit(X_train, y_train)
        # Calcolo delle accuracy per train e test
        accuracy_train.append(dtc.score(X_train, y_train))
        accuracy_test.append(dtc.score(X_test, y_test))

    # Creazione del grafico per min_samples_leaf
    plt.plot(min_samples_leaf_values, accuracy_train[:len(min_samples_leaf_values)], label='Train', color='purple')
    plt.plot(min_samples_leaf_values, accuracy_test[:len(min_samples_leaf_values)], label='Test', color='green')

    plt.title('MIN_SAMPLES_LEAF')
    plt.xlabel('Min_samples_leaf')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(5, 31, 5))
    plt.show()

    # Cross-validation su training set con un valore fisso di max_depth (es. 5)
    dtc_cv = DecisionTreeClassifier(max_depth=5, random_state=42)
    cv_scores = cross_val_score(dtc_cv, X_train, y_train, cv=kf, scoring='accuracy')
    
    # Salvataggio dei risultati di cross-validation
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())
    
    # Predizione sul test set con max_depth = 5
    dtc_cv.fit(X_train, y_train)
    y_pred = dtc_cv.predict(X_test)

# Visualizzazione dei risultati delle cross-validation
for run in range(n_runs):
    print(f"Run {run + 1}:")
    print(f"  Cross-validation accuracy mean: {cv_means[run]:.4f}")
    print(f"  Cross-validation accuracy std: {cv_stds[run]:.4f}")

# Calcolo della media di tutte le medie dei run
overall_mean = np.mean(cv_means)
print(f"---------------------------------------------------------------------------------")
print(f"\nTOTAL MEAN: {overall_mean:.4f}")

# Creazione dei grafici
# Grafico delle accuratezze medie della cross-validation
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, n_runs+1), cv_means, yerr=cv_stds, fmt='-o', color='green', label='CV Mean Accuracy')
plt.title('Cross-Validation Accuracy (Mean and Std) for Each Run of Decision Tree')
plt.xlabel('Run')
plt.ylabel('CV Accuracy (Mean ± Std)')
plt.legend()
plt.grid(True)
plt.show()
