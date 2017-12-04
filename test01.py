# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:59:48 2017

@author: Think
"""

import numpy as np
import csv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image



#%% Generate sample data
#centers= [[1,1], [0.5,3], [1.5,-1], [-1,6],[-3,8.5]]
#X, _= make_blobs(n_samples=12000, centers=centers, cluster_std=0.6)
X=np.zeros((1337,2))
csv_file=csv.reader(open('numbers.txt','r'))
#print(csv_file)
k=-1
for stu in csv_file:
    k=k+1
    X[k,0]=float(stu[0])
    X[k,1]=float(stu[1])

 # The bandwidth can be automatically estimated
bandwidth= estimate_bandwidth(X, quantile=.1, n_samples=500)
ms= MeanShift(bandwidth=bandwidth/3, bin_seeding=True)
ms.fit(X)
labels= ms.labels_
cluster_centers= ms.cluster_centers_
 
n_clusters_= labels.max()+1
 
#%% Plot result
plt.figure(1)
plt.clf()

colors= cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members= labels == k
    cluster_center= cluster_centers[k]
    plt.plot(X[my_members,0], X[my_members,1], col+ '.')
    
    plt.plot(cluster_center[0], cluster_center[1],
             'o', markerfacecolor=col,
             markeredgecolor='w',
 markersize=8)
plt.title('Estimated number of clusters: %d' % 
n_clusters_)
plt.show()