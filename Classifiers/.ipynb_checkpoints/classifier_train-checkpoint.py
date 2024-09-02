import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import ydata_profiling

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,fbeta_score,confusion_matrix,classification_report,precision_score,recall_score

import warnings
warnings.filterwarnings('ignore')

#carico il dataset 
df=pd.read_csv("C:/Users/fracu/OneDrive - Università degli Studi di Bari/Desktop/ICON-DIABETE/Dataset/diabetes.csv")
df.head()

#elimino eventuali duplicati
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


#divido le feature dai risultati
X=df.drop('Outcome',axis=1)
y=df['Outcome']

#divisione dei dati in train e test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,shuffle=True,random_state=42)


#istogramma per vedere come sono distribuiti i dati
print('-'*40+'REPRESENTATION OF THE DISTRIBUTION OF DATA'+'-'*40)
df.hist(figsize = (10,10),color='purple')
plt.show()

#grafico per visualizzare i dati che incidono maggiormente del dataset
print('-'*40+'REPRESENTATION OF THE MOST RELEVANT DATA'+'-'*40)
variables = ['Pregnancies', 'Glucose', 'Blood_Pressure', 'Skin_Thickness', 'Insulin', 'BMI', 'Diabetes_Pedigree_Function', 'Age']

plt.figure(figsize=(15, 10))

for i, var in enumerate(variables, 1):
    plt.subplot(2, 4, i)
    plt.hist(df.loc[df['Outcome']==0, var], bins=30, alpha=0.5, label='Outcome=0', color='purple')
    plt.hist(df.loc[df['Outcome']==1, var], bins=30, alpha=0.5, label='Outcome=1', color='green')
    
    plt.title(f'{var} based on Outcome')
    plt.legend()
    
plt.tight_layout()
plt.show()

#eseguo diversi modelli di classificatori
print('-'*40+'DIFFERENT MODELS OF CLASSIFIER'+'-'*40)

#KNN CLASSIFIER
knn=KNeighborsClassifier()
#addestramento di X,y train
knn.fit(X_train,y_train)
#predict dei dati di test X
y_predict_knn=knn.predict(X_test)

print("K Nearest Neighbors Classification: ")
print(classification_report(y_test,y_predict_knn))

#eseguo il calcolo della f2 score del knn
f2_knn=fbeta_score(y_test,y_predict_knn,beta=2)
print(f"F2 Score for KNN Classifier: {f2_knn}\n")

#DECISION TREE CLASSIFIER
dtc=DecisionTreeClassifier()
#addestramento di X,y train
dtc.fit(X_train,y_train)
#predict dei dati di test X
y_predict_dtc=dtc.predict(X_test)

print("Decision Tree Classification: ")
print(classification_report(y_test,y_predict_dtc))

#f2 score per il decision tree
f2_dtc=fbeta_score(y_test,y_predict_dtc,beta=2)
print(f"F2 Score for Decision Tree Classifier: {f2_dtc}\n")


#RANDOM FOREST CLASSIFIER
rf=RandomForestClassifier()
#addestramento X,y train
rf.fit(X_train,y_train)
#predict
y_predict_rf=rf.predict(X_test)

print("Random Forest Classification: ")
print(classification_report(y_test,y_predict_rf))

#f2score per il random forest
f2_rf=fbeta_score(y_test,y_predict_rf,beta=2)
print(f"F2 Score for Random Forest Classifier: {f2_rf}\n")

#ADA BOOST CLASSIFIER
#creazione di un classificatore decisional tree base 
dtc_base=DecisionTreeClassifier(max_depth=1)
#creo il classificatore ada boost con il dtc_base
ada=AdaBoostClassifier(dtc_base,n_estimators=50)
#addestramentro X,y train per ada
ada.fit(X_train,y_train)
#predict
y_predict_ada=ada.predict(X_test)

print("AdaBoost Classification: ")
print(classification_report(y_test,y_predict_ada))

#f2Score per ada 
f2_ada=fbeta_score(y_test,y_predict_ada,beta=2)
print(f"F2 Score for AdaBoost Classifier: {f2_ada}\n")

#XGBOOST CLASSIFIER
xgb=XGBClassifier()
#addestramento per X,y train
xgb.fit(X_train,y_train)
#predict
y_predict_xgb=xgb.predict(X_test)

print("XGBoost Classification: ")
print(classification_report(y_test,y_predict_xgb))

#f2Score per xgb
f2_xgb=fbeta_score(y_test,y_predict_xgb,beta=2)
print(f"F2 Score for XGBoost Classifier: {f2_xgb}\n")

