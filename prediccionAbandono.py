# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:49:54 2021

@author: Fede
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv("telco_churn.csv")
df.head()

df.isna().sum().sum() #missing values in the data set 
                        # si da 0 There is no missing value in the data set so we can jump to explore it
                        
df.columns
df.dtypes

df.Churn.value_counts()
# chrun que es la y, lo q queremos predecir
# esta desbalanceada hay muchos mas NO que si
# hay q tener cuidado

columns = df.columns
binary_cols = []

for col in columns:
    if df[col].value_counts().shape[0] == 2:
        binary_cols.append(col)
        

print(binary_cols) #estas son categorias o caracteristicas
        #que son 2 opciones, el resto mas de 2
# Categorical features with multiple classes
multiple_cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']


#%% Ahora tenemos separada la data. Empezamos con la binaria
#Graficamos la relacion entre las 2 opciones de cada una de estas categorias
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("gender", data=df, ax=axes[0,0])
sns.countplot("SeniorCitizen", data=df, ax=axes[0,1])
sns.countplot("Partner", data=df, ax=axes[0,2])
sns.countplot("Dependents", data=df, ax=axes[1,0])
sns.countplot("PhoneService", data=df, ax=axes[1,1])
sns.countplot("PaperlessBilling", data=df, ax=axes[1,2])

# Se puede observar en estos graficos que la mayoria tiene phone service
# y que la mayoria de los customers son senior
# Pero que relacion tienen estas caracteristicas con el churn?

churn_numeric = {'Yes':1, 'No':0}
df.Churn.replace(churn_numeric, inplace=True)

# Despues de ponerle valores numericos al churn
# voy a relacinarlo con cada caracterisitca binaria
df[['gender','Churn']].groupby(['gender']).mean()
# Por pantalla se ve que el promedio  para female o male es lo mismo
# practicamente entonces no me sirve para categorizar el churn

df[['SeniorCitizen','Churn']].groupby(['SeniorCitizen']).mean()
df[['Partner','Churn']].groupby(['Partner']).mean()
df[['Dependents','Churn']].groupby(['Dependents']).mean()
df[['PhoneService','Churn']].groupby(['PhoneService']).mean()
df[['PaperlessBilling','Churn']].groupby(['PaperlessBilling']).mean()

# el resto si hay diferencia aunq phone service sea un 2% la pondre igual
# se pueden escribir Tablas como la siguient pandas-pivot_table
table = pd.pivot_table(df, values='Churn', index=['gender'],
                    columns=['SeniorCitizen'], aggfunc=np.mean)
print(table)





#%% Ahora analizaremos la data no binaria

sns.countplot("InternetService", data=df)
#la vinculamos con churn para ver que tanto influyen en la salida
df[['InternetService','Churn']].groupby('InternetService').mean()
# Aca si se observa que no tener internet service no modifica abandonar
# y q fibra optica es bastante comparado con dsl
# Pensando es claro decir que tiene mala conexion de fibra optica
# El servicio de internet es clave para mantener clinetes?

df[['InternetService','MonthlyCharges']].groupby('InternetService').mean()
# Fibra optica es mucho mas caro, entonces ahi es capaz el abandono que sea caro
# no mala calidad, no se sabe


#Los servicios que da internet
# La tercer opcion es q no tiene servicio de internet eso podriamos eliminar de
# cierta manera info
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("StreamingTV", data=df, ax=axes[0,0])
sns.countplot("StreamingMovies", data=df, ax=axes[0,1])
sns.countplot("OnlineSecurity", data=df, ax=axes[0,2])
sns.countplot("OnlineBackup", data=df, ax=axes[1,0])
sns.countplot("DeviceProtection", data=df, ax=axes[1,1])
sns.countplot("TechSupport", data=df, ax=axes[1,2])

df[['StreamingTV','Churn']].groupby('StreamingTV').mean()
df[['StreamingMovies','Churn']].groupby('StreamingMovies').mean()
df[['OnlineSecurity','Churn']].groupby('OnlineSecurity').mean()
df[['OnlineBackup','Churn']].groupby('OnlineBackup').mean()
df[['DeviceProtection','Churn']].groupby('DeviceProtection').mean()
df[['TechSupport','Churn']].groupby('TechSupport').mean()
#Todos tienen un rate distinto

df.PhoneService.value_counts()
#hay muchos yes vs pocos no
df.MultipleLines.value_counts()
# para tener multipleLines hay q tener phoneservice
#entonces eliminaremos phoneservice, solo quedara NOTIENEPHONESERVICE
# cuanto categorizemos en 1 0 0
df[['MultipleLines','Churn']].groupby('MultipleLines').mean()



# Ahora analizamos Contract, payment method
sns.countplot("Contract", data=df)
df[['Contract','Churn']].groupby('Contract').mean()
# aca queda demostrado que los que son en short term
# tienden a irse mas que los clientes anulaes
sns.countplot("PaymentMethod", data=df)
df[['PaymentMethod','Churn']].groupby('PaymentMethod').mean()
# los que pagan con electronic check son los q mas se van pero tambien son
# los que mas hay

#%% Ahora analizaremos las datas continuas
# las caracteristicas continuas son: tenure, monthlycharges y totalCharges
# Pero total es al pedo usarla xq depende de las otras 2
# es reduntante y no hay q overfiting

fig, axes = plt.subplots(1,2, figsize=(12, 7))

sns.distplot(df["tenure"], ax=axes[0])
sns.distplot(df["MonthlyCharges"], ax=axes[1])

df[['tenure','MonthlyCharges','Churn']].groupby('Churn').mean()
# Aca se puede ver que si estas mas tiempo(es decir tu mean es mayor)
# no abandonas el servicio

# si tenes un contract largo es q no te vas y seguro son tenure largo
df[['Contract','tenure']].groupby('Contract').mean()
# La relacion es obvia entonces se puede ver que contracts
# me da info redundante xq se va a ir igual con tenure dicen lo mismo 
# pero relacionan con toras caracteristicas
#%% Elimino Datos redundates
# 1) Customer ID 2) Gender 3) PhoneService 4) Contract 5) TotalCharges
df.drop(['customerID','gender','PhoneService','Contract','TotalCharges'], axis=1, inplace=True)

### Empiezo a procesar la data
#1. Lo categorico debe de ser 1 o 0
#2. Las continuas tmb si no un 70 es mucho mas imponente en la red
#3.  La parte de bueno malo regular...

# one hot encoder creo q es getdummies de pandas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# categoricas no binarias
cat_features = ['SeniorCitizen', 'Partner', 'Dependents',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
X = pd.get_dummies(df, columns=cat_features, drop_first=True)

#los escalo a los continuos de 0 a 1
sc = MinMaxScaler()
a = sc.fit_transform(df[['tenure']])
b = sc.fit_transform(df[['MonthlyCharges']])

X['tenure'] = a
X['MonthlyCharges'] = b
# Luego a X le agregue 2 columnas, estoy preparando la entrada de la red

# Ahora hay q reacomodar la diferencia que hay entre los NO y SI, su cantidad
# en churn 
# Para eso upsampling que significa incrementar el numeros de samples de la clae
# que menos tiene eligiendo filas aleatorias

sns.countplot('Churn', data=df).set_title('Class Distribution Before Resampling')
X_no = X[X.Churn == 0]
X_yes = X[X.Churn == 1]

print(len(X_no),len(X_yes))

X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)
print(len(X_yes_upsampled))

X_upsampled = X_no.append(X_yes_upsampled).reset_index(drop=True)

sns.countplot('Churn', data=X_upsampled).set_title('Class Distribution After Resampling')
# lo que se hizo primero fue dividir la data entre los que churn yes y churn no
# ahi vemos la diferencia de len 
# entonces igualamos la cantidad de 5mil y pico repitiendo aleatoriamente casos
# en los cuales churn yes

#%% RED MODEL

from sklearn.model_selection import train_test_split

X = X_upsampled.drop(['Churn'], axis=1) #features (independent variables)
y = X_upsampled['Churn'] #target (dependent variable)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Hay 4 redes para probar y ver cual es la mejor posible
# Ridge Classifier (?)
# Tree y con gridsearch
# ANN 


from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

clf_ridge = RidgeClassifier() #create a ridge classifier object
clf_ridge.fit(X_train, y_train) #train the model

pred = clf_ridge.predict(X_train)  #make predictions on training set
accuracy_score(y_train, pred) #accuracy on training set

confusion_matrix(y_train, pred)

pred_test = clf_ridge.predict(X_test)
accuracy_score(y_test, pred_test)
# Mostro una accuracy de 75 en train 76 en test  necesitamos mas

# Tree{}

from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
clf_forest.fit(X_train, y_train)

pred = clf_forest.predict(X_train)
accuracy_score(y_train, pred)

confusion_matrix(y_train, pred)

pred_test = clf_forest.predict(X_test)
accuracy_score(y_test, pred_test)

#88% en train 84 en test esto indica que hay 4% mas en train
# POR LO TANTO HAY OVERFITING 
# Aca se puede ver el problema de distintas maneras xq es un arbol
# podemos cambiar las depth xq mientras mas en las raices vaamos generaliza peor
# pero si disminuimos el depth perderemos accuracy
# entonces hay q tener cuidado que parametros modificamos para mejorar la red
# podriamos aumentar los trees..

# LA q vamos a hacer es cross Validation 
# utilizando GridSearchCV We can both do cross-validation and try different parameters using GridSearchCV.

from sklearn.model_selection import GridSearchCV #CV = cross Validation

parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}
forest = RandomForestClassifier()
clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)

# cv = 5 means having a 5-fold cross validation (So dataset is divided into 5 subset.)
#At each iteration, 4 subsets are used in training and the other subset is used as test set
# When 5 iteration completed, the model used all samples as both training and test samples

# n_jobs parameter is used to select how many processors to use. -1 means using all processors.

clf.fit(X, y)

clf.best_params_
clf.best_score_


#%% ANN


import tensorflow as tf
from tensorflow import keras


# Voy a crear una red entonces al escribir keras.sequential
# es como que arranco la red y sus layers las voy poniendo y asi armando
# la red.
# primerp IMput layer que lleve la misma cantidad de neuronas que de colum de x
# va a ser una dense
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(25,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
    
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)


model.evaluate(X_test,y_test)


yp = model.predict(X_test)
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
#la red tiene 80% asi q erra 

from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test, y_pred))

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Thuth')













