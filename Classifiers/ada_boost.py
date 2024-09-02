import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Caricamento del dataset
df = pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di Bari/Desktop/ICON-DIABETE/Dataset/diabetes.csv")

# Rimozione dei duplicati
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

# Separazione delle features e del target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Divisione del dataset in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

# Definizione dei valori da esplorare per n_estimators (da 100 a 1000, con incrementi di 100)
n_values_estimators = range(100, 1001, 100)

# Definizione dei valori da esplorare per learning_rate
learning_rates = np.arange(0.1,1.01,0.1)

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

# Addestramento del modello AdaBoost con i nuovi parametri ottimizzati
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
ada.fit(X_train, y_train)

# Valutazione dell'accuratezza con cross-validation
accuracy = cross_val_score(ada, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy: ", accuracy)
print("\nMean accuracy: ", accuracy.mean())
print("\nStandard accuracy: ", accuracy.std())

# Valutazione del modello sui dati nel set di test
y_predict_ada = ada.predict(X_test)
accuracy_test = accuracy_score(y_test, y_predict_ada)
print("Accuracy on test set: ", accuracy_test)

print("Classification report: ")
print(classification_report(y_test, y_predict_ada))

# Creazione della confusion matrix
plt.figure(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_predict_ada), annot=True, fmt='d', cmap='plasma')
plt.title('CONFUSION MATRIX', fontweight='bold')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()
