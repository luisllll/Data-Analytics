# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:57:23 2022

@author: luisl
"""

import os

os.getcwd()

#change working directory

os.chdir('C:/Users/luisl/OneDrive/Escritorio/ml-scripts')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.cluster import KMeans


 #import csv
 cluster_credit_card = pd.read_csv("C:/Users/luisl/OneDrive/Escritorio/ml-scripts/credit card clean.csv") # creates a Pandas data frame for credit card dataset
 
 
 
 print(cluster_credit_card)
 
 
 #show columns
 cluster_credit_card.columns
 
  # show data types
  cluster_credit_card.dtypes
  
  
#aplicar un k-means
distortions = []
K=range(1,10)
for k in K:
    kmeanModel=KMeans(n_clusters=k)
    kmeanModel.fit(cluster_credit_card)
    distortions.append(kmeanModel.inertia_)


print(kmeanModel.inertia_)#como de bien ajusta el modelo


#plotear resilktado
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')#con puntitos en los puntod
plt.xlabel('k')
plt.ylabel('Distortion score')
plt.title('The Elbow Method showing the optimal k')
plt.show()




#Determinar el k optimo con el shiloutte score

from sklearn.metrics import silhouette_score


n_clusters= list(range(2,10))
print ("Number of clusters from 2 to 9: \n", n_clusters)

for n in n_clusters:
    clusterer=KMeans(n_clusters=n).fit(cluster_credit_card)
    preds=clusterer.predict(cluster_credit_card)
    centers=clusterer.cluster_centers_
    score=silhouette_score(cluster_credit_card,preds)
    print("For n={}, sil score={})".format(n,score))








