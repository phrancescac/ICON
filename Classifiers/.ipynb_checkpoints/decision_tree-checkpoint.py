import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ydata_profiling

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

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

#esplorazione dei valori della max_depth dell'albero
max_depth_values=range(1,16)

#definisco due liste sia per il train che per il test dove memorizzare le rispettive accuracy
accuracy_train=[]
accuracy_test=[]

#ciclo su ogni valore dell'albero del max_depth
for depth in max_depth_values:
    #creo il classificatore
    dtc=DecisionTreeClassifier(max_depth=depth,random_state=42)
    #addestramento per X,y train
    dtc.fit(X_train,y_train)
    #calcolo i valori di accuracy sia per il train che per il test
    accuracy_train.append(dtc.score(X_train,y_train))
    accuracy_test.append(dtc.score(X_test,y_test))

#visualizzo un grafico che inserisce le accuracy in funzione dei valori di max_depth
plt.plot(max_depth_values,accuracy_train,label='Train',color='purple')
plt.plot(max_depth_values,accuracy_test,label='Test',color='green')

#creazione grafico per il max_depth
#imposto il titolo del grafico e i nomi degli assi (max depht, accuracy)
plt.title('MAX_DEPTH')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend()#legenda del grafico
plt.grid(True)#griglia attiva
plt.xticks(range(1,16))#ticks sull'asse x
plt.show()#mostra il grafico

#esplorazione del mean_samples_leaf dell'albero 
min_samples_leaf=range(5,31,5)

#creo anche qui due liste per train e test dove inserire le relative accuracy
accuracy_train=[]
accuracy_test=[]

#ciclo su ogni valore del min_samples_leaf
for leaf in min_samples_leaf:
    #creo il classificatore
    dtc=DecisionTreeClassifier(min_samples_leaf=leaf,random_state=42)
    #addestramento sia per X,y train
    dtc.fit(X_train,y_train)
    #calcolo le relative accuracy per train e test
    accuracy_train.append(dtc.score(X_train,y_train))
    accuracy_test.append(dtc.score(X_test,y_test))

#creo un nuovo grafico dove inserisco le accuracy dei valori del min_samples_leaf
plt.plot(min_samples_leaf,accuracy_train,label='Train',color='purple')
plt.plot(min_samples_leaf,accuracy_test,label='Test',color='green')

#creazione grafico per il min_samples_leaf
#imposto il titolo del grafico e degli assi x e y
plt.title('MIN_SAMPLES_LEAF')
plt.xlabel('Min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(5,31,5))
plt.show()

#addestro nuovamente il modello di decision tree con i nuovi parametri ottimizzati
dtc=DecisionTreeClassifier(min_samples_leaf=25,random_state=42)
dtc.fit(X_train,y_train)

#valutazione accuracy del modello 
accuracy=cross_val_score(dtc,X_train,y_train,cv=5,scoring='accuracy')
print("Cross-validation accuracy: ",accuracy)
print("\nMean accuracy: ",accuracy.mean())
print("\nStandard accuracy: ",accuracy.std())

#valutazione del model sui dati 
y_pred_dtc=dtc.predict(X_test)
accuracy=accuracy_score(y_test,y_pred_dtc)
print("Accuracy on test set: ",accuracy)

print("Classification report: ")
print(classification_report(y_test,y_pred_dtc))

#creazione della confusion matrix tra i valori reali e quelli predetti
plt.figure(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test,y_pred_dtc),annot=True,fmt='d',cmap='plasma')
plt.title('CONFUSION MATRIX')
plt.xlabel('Predict Class')
plt.ylabel('True Class')
plt.show()
