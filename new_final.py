# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:25:31 2020

@author: vaibhav
"""

import numpy as np
import pandas as pd
    
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
Normalizer = MinMaxScaler(feature_range = (0, 1))
X = Normalizer.fit_transform(X)

#Training the SOM
from minisom import MiniSom

som = MiniSom(x= 15,y =15,
              sigma=0.8,
              learning_rate=0.5,
              input_len = 14)

som.random_weights_init(X)

som.train_random(data = X, num_iteration = 200)
  
#Visualizing the Results
from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
for i, j in enumerate(X):    
  if Y[i] == 1:
    W_node = som.winner(j)
    plot(W_node[0] + 0.5,
         W_node[1] + 0.5,
         'og',markerfacecolor = 'none',
         markersize = 9,markeredgewidth = 1)
  else:
    W_node = som.winner(j)
    plot(W_node[0] + 0.5,
         W_node[1] + 0.5,
         'xr')
show()
  
#finding fraud
threshold_value= 0.8
map_of_distance = som.distance_map().T

exceed_map=[]
i = 0
j = 0
for i in range(0,15):
  for j in range(0,15):
    if map_of_distance[i,j] >= threshold_value:
      add_map=(i,j)
      exceed_map.append(add_map); #all the id which are greater then threshold value
mappings = som.win_map(X) #dictionary of all the data 
print(exceed_map)

i = 0
outlier_add = []
for i in range(len(exceed_map)):
  map_index = tuple(exceed_map[i])
  outlier_add.append(mappings[map_index]); # get the tuples off the outier in the exceed map list 
#print(outlier_add)

#remove empty set from outlier
while [] in outlier_add:
  outlier_add.remove([]);
  
i = 0
outlier = np.zeros((1,14))
  
for i in range(len(outlier_add)):
  outlier = np.concatenate((outlier,outlier_add[i]), axis = 0) # add everything in outlier 
  
outlier = outlier[1:,:]
outlier = Normalizer.inverse_transform(outlier)
#print(outlier)

i = 0
j = 0
X_with_id = dataset.iloc[:,0:1].values #only id
#print(X_with_id)
X_turn_back = Normalizer.inverse_transform(X)
X_with_id = np.hstack((X_with_id, X_turn_back))
#print(X_with_id)

result = np.zeros((1,15))

for j in range(len(outlier)):
    for i in range(len(X)):
        if outlier[j,0] == X_with_id[i,1] and\
        outlier[j,1] == X_with_id[i,2] and\
        outlier[j,2] == X_with_id[i,3] and\
        outlier[j,3] == X_with_id[i,4] and\
        outlier[j,4] == X_with_id[i,5] and\
        outlier[j,5] == X_with_id[i,6] and\
        outlier[j,6] == X_with_id[i,7] and\
        outlier[j,7] == X_with_id[i,8] and\
        outlier[j,8] == X_with_id[i,9] and\
        outlier[j,9] == X_with_id[i,10] and\
        outlier[j,10] == X_with_id[i,11] and\
        outlier[j,11] == X_with_id[i,12] and\
        outlier[j,12] == X_with_id[i,13] and\
        outlier[j,13] == X_with_id[i,14]:
            result = np.vstack((result, X_with_id[i]));
result = result[1:,:]
print(result[:,0])
