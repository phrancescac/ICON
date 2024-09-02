import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ydata_profiling

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

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

#definisco i valori da esplorare per n_estimators(da 100 a 1000,ogni 100)
n_values_estimators=range(100,1001,100)

#definisco i valori da esplorare per max_depth
max_depth_values=range(1,11)

#creo un subplot per visualizzare grafici con più valori delle accuratezze
fig,axes=plt.subplots(5,2,figsize=(15,30))

for i, n_estimators in enumerate(n_values_estimators):
    train_accuracy = []
    test_accuracy = []

    for max_depth in max_depth_values:
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        rfc.fit(X_train, y_train)
        train_pred = rfc.predict(X_train)
        test_pred = rfc.predict(X_test)

        train_accuracy.append(accuracy_score(y_train, train_pred))  # aggiunta dell'accuratezza di addestramento per ogni max_depth
        test_accuracy.append(accuracy_score(y_test, test_pred))
        
    # axes
    ax = axes[i//2, i%2]

    ax.plot(max_depth_values, train_accuracy, label=f'Train Accuracy, n_estimators={n_estimators}',color='purple')
    ax.plot(max_depth_values, test_accuracy, label=f'Test Accuracy, n_estimators={n_estimators}',color='green')

    ax.set_title(f'n_estimators={n_estimators}', fontweight='bold')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(1, 11, 1))

plt.tight_layout()
plt.show()

#addestramento del modello rf con i nuovi parametri ottimizzati
rf=RandomForestClassifier(n_estimators=500,max_depth=1,random_state=42)
rf.fit(X_train,y_train)

#valutazione accuracy con cross-validation
accuracy=cross_val_score(rf,X_train,y_train,cv=5,scoring='accuracy')
print("Cross-validation accuracy: ",accuracy)
print("\nMean accuracy: ",accuracy.mean())
print("\nStandard accuracy: ",accuracy.std())

#valutazione del modello sui dati nel set
y_predict_rf=rf.predict(X_test)
accuracy_test=accuracy_score(y_test,y_predict_rf)
print("Accuracy on test set: ",accuracy_test)

print("Classification report: ")
print(classification_report(y_test,y_predict_rf))

#creazione della confusion matrix
plt.figure(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test,y_predict_rf),annot=True,fmt='d',cmap='plasma')
plt.title('CONFUSION MATRIX',fontweight='bold')
plt.xlabel('Predict Class')
plt.ylabel('True Class')
plt.show()