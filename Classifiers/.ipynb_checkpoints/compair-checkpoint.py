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

#creo una lista con tutti i classificatori utilizzati e che voglio confrontare
classifiers={
    'KNN': KNeighborsClassifier(),
    'DECISION TREE CLASSIFIER': DecisionTreeClassifier(),
    'RANDOM FOREST CLASSIFIER': RandomForestClassifier(),
    'ADABOOST CLASSIFIER':AdaBoostClassifier(),
    'XGBOOST CLASSIFIER': XGBClassifier()
}

#calcolo il valore di accuracy
accuracy={}

for name,classifier in classifiers.items():
    scores=cross_val_score(classifier,X,y,cv=5,scoring='accuracy')
    accuracy[name]=scores.mean()
    
#creazione del grafico a barre con le accuracy medie ottenute
plt.figure(figsize=(10,6))
plt.bar(accuracy.keys(),accuracy.values(),color='purple')
plt.title('ACCURACY SCORES OF DIFFERENT CLASSIFIERS',fontweight='bold')
plt.xlabel('Classifiers')
plt.ylabel('Mean accuracy')
plt.ylim(0.6,1.0)
plt.xticks(rotation=45)
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()

