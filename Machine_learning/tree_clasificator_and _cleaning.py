# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:58:19 2022

@author: luisl
"""


import os
os.chdir('C:/Users/luisl/OneDrive/Escritorio/ml-scripts')

# Let's start off by importing the relevant libraries
import pandas as pd
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


#importar csv
# Import training and test sets into the scripts
raw_training_df = pd.read_csv("C:/Users/luisl/OneDrive/Escritorio/ml-scripts/train.csv") # creates a Pandas data frame for training set
raw_test_df  = pd.read_csv("C:/Users/luisl/OneDrive/Escritorio/ml-scripts/test.csv") # similar


#Explore data
print(raw_test_df)
print(raw_training_df)

raw_training_df.dtypes
raw_test_df.dtypes


#contar resulkatdos dentro de la variable Survived
count_classes = pd.value_counts(raw_training_df['Survived'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Class Label Histogram")
plt.xlabel("Class")
plt.ylabel("Frequency");




##PROCESADO DE DATOS


#elimnar columnas innecesarias
training_df = raw_training_df.drop(['Name', 'Ticket'], axis=1)

#rellenas nans de age con la media
training_df['Age']=training_df['Age'].fillna(math.ceil(training_df['Age'].mean()))


#para alimentar el arbol de clasificación necesitamos hacer one hot encoding de las strings

def encode_target(df,target_column):
    df_mod=df.copy()
    targets=df_mod[target_column].unique()#saca valores unicos
    map_to_int={name:n for n,name in enumerate(targets)}#enumerar lista y crea un diccionario
    df_mod[target_column]=df_mod[target_column].replace(map_to_int)
    
    return(df_mod,targets)


training_df, sex_targets=encode_target(training_df,"Sex")
training_df, embarked_targets = encode_target(training_df,"Embarked")
training_df, cabin_targets = encode_target(training_df,"Cabin")

training_df.dtypes 


#crear train(sin la target)
X = training_df.loc[:, training_df.columns != 'Survived']

#crear variable target
y = training_df.loc[:, training_df.columns == 'Survived']  
    
    
#CV 70/30
X_train, X_CV, y_train, y_CV = train_test_split(X,y,test_size = 0.3, random_state = 0) 
    
    
print(X_train)

#decisiontree
dt= DecisionTreeClassifier(min_samples_split=20,random_state=99)
    
    
#entrenar modelo
dt.fit(X_train, y_train)
    
#predecir lara X_CV   
y_pred=dt.predict(X_CV)  
    
    
    

  #PLOT DDE LA MATRIZ DE CONFUSION  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    
    
    

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_CV, y_pred)
np.set_printoptions(precision=2)
    
    
 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='CM')
plt.show()
    
    


#metricas de precison

prec = cnf_matrix[1,1] / (cnf_matrix[0,1] + cnf_matrix[1,1])
print("The precision of the ML model is ", round(prec, 3))

recl = cnf_matrix[1,1] / (cnf_matrix[1,0] + cnf_matrix[1,1])
print("The recall of the ML model is ", round(recl, 3))

f1 = 2*((recl*prec)/(recl+prec))
print("The f1-score of the ML model is %f." % round(f1, 3))

acc = (cnf_matrix[1,1] + cnf_matrix[0,0]) / ((cnf_matrix[0,1] + cnf_matrix[1,1]) + cnf_matrix[0,0] + cnf_matrix[1,0])
print("The accuracy of the ML model is ", round(acc, 3))
