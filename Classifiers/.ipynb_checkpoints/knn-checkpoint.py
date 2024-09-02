
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ydata_profiling

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import fbeta_score,accuracy_score,confusion_matrix,classification_report,precision_score,recall_score

import warnings
warnings.filterwarnings('ignore')

#carico il dataset
df=pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di Bari/Desktop/ICON-DIABETE/Dataset/diabetes.csv")
df.head()

#elimino i duplicati
df=df.drop_duplicates()

#elimina valori nulli nel dataset
for col in ["Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI"]:
    mean = df[df[col] != 0][col].mean()
    std = df[df[col] != 0][col].std()

    values = df[col].values

    np.random.seed(23)

    for i, val in enumerate(values):
        if val == 0:
            values[i] = mean + std * (np.random.rand() * 2 - 1) * 1.2
    df[col] = pd.Series(values).astype(df[col].dtype)

X=df.drop('Outcome',axis=1)
y=df['Outcome']

#divido il dataset in train e test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,shuffle=True,random_state=42)


# Creazione di un oggetto StandardScaler per standardizzare le features numeriche
scaler = StandardScaler()

# Copia del dataframe originale per applicare la standardizzazione
df_scaler = df.copy()

# Selezionare le features numeriche da standardizzare
num_features = ['Pregnancies', 'Glucose', 'Blood_Pressure', 'Skin_Thickness', 'Insulin','BMI','Diabetes_Pedigree_Function','Age']

# Standardizzazione delle features numeriche
scaled_features = scaler.fit_transform(df_scaler[num_features])
df_scaler[num_features] = scaled_features

#seleziono le features e il target
X=df_scaler.drop('Outcome',axis=1)#features
y=df_scaler['Outcome']#valori output 

#divido il nuovo dataset in train e test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,shuffle=True,random_state=42)

# Esplorazione dei valori di n_neighbors per KNN
n_neighbors_values = range(1, 30)

# Definisco due liste per memorizzare le accuracy di train e test
accuracy_train_knn = []
accuracy_test_knn = []

# Ciclo su ogni valore di n_neighbors
for neighbors in n_neighbors_values:
    # Creo il classificatore KNN
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    # Addestramento sul training set
    knn.fit(X_train, y_train)
    # Calcolo delle accuracy per train e test
    accuracy_train_knn.append(knn.score(X_train, y_train))
    accuracy_test_knn.append(knn.score(X_test, y_test))

# Creazione del grafico per n_neighbors
plt.plot(n_neighbors_values, accuracy_train_knn, label='Train', color='purple')
plt.plot(n_neighbors_values, accuracy_test_knn, label='Test', color='green')

# Impostazione del titolo e delle etichette degli assi
plt.title('KNN - Number of Neighbors')
plt.xlabel('Number of Neighbors (n_neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, 30))
plt.show()

#creo il classificatore knn
knn=KNeighborsClassifier()
#addestramento per X,y train sul training set
knn.fit(X_train,y_train)

#valutazione l'accuracy del modello usandp la cross-validation 
accuracy=cross_val_score(knn,X_train,y_train,cv=5,scoring='accuracy')

print("Cross-validation accuracy: ",accuracy)
print("\nMean accuracy: ",accuracy.mean())
print("\nStandard deviation of accuracy: ",accuracy.std())

#predict e addestramento valutazione  accuracy del modello knn
knn=KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train,y_train)
y_predict_knn=knn.predict(X_test)

accuracy_test=accuracy_score(y_test,y_predict_knn)
print("Accuracy with k neighbors=29: ",accuracy_test)

print("Classification report: ")
print(classification_report(y_test,y_predict_knn))

#creazione della confusion matrix
plt.figure(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test,y_predict_knn),annot=True,fmt='d',cmap='plasma')
plt.title('CONFUSION MATRIX',fontweight='bold')
plt.xlabel('Predict Class')
plt.ylabel('True Class')
plt.show()
