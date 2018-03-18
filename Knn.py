#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the third assignment in class Methods in Bioinformatics and refers to
Classification alorithms & in particular the K-nn and Naive Bayes.

This particular script contains the implementation of K nearest neighbours
classifier.

"""

#==============================================================================
#   Libraries:
#==============================================================================

from Functions import*
from scipy.spatial import distance
#import numpy as np

def Knn(Data,Train_Matrix,Test_Matrix,Stack_Matrix,Train_labels,k ,Metric):

	
	
	X = Data.T;	N= X.shape[1]
		
	n = Test_Matrix.shape[1]
	m = Train_Matrix.shape[1]

	Predictions = []
	
	for i in range(n):
		vec_01 = Test_Matrix[:,i]
		Distances = np.zeros(m)
	
		for j in range(m):

			vec_02 = Train_Matrix [:,j]

			Distances[j] = dist_function(Stack_Matrix,Metric,vec_01,vec_02)

		Sorted_idx = np.argsort(Distances)
		k_indices  = Sorted_idx[0:k]

		k_labels  = list(Train_labels[k_indices])

		predict_class = max(k_labels,key=k_labels.count)
		
		Predictions.append(predict_class)

	return(Predictions)




#==============================================================================
#  Basic Metric Distances Functions:
#==============================================================================


def euclidean_distance(x, y):
	return np.sqrt(np.sum((x-y)**2))

def Manhattan_Distance(x,y):
	return np.sum(np.abs(x-y))

def Mahalanobis_Distance(Arr,x,y):
	D = Arr.shape[0]
	S = np.cov(Arr) 
	return np.sqrt((x-y).reshape(1,D).dot(np.linalg.inv(S)).dot((x-y).reshape(D,1)))



#==============================================================================
#  Change distance function here according to needs:
#==============================================================================


def dist_function(Arr,Distance,x, y):
	if Distance == 'a':
		return euclidean_distance(x, y)
	elif Distance == 'b':
		return Manhattan_Distance(x,y)
	elif Distance == 'c':
		return Mahalanobis_Distance(Arr,x,y)
	elif Distance == 'd':
		D = x.shape[0]
		return distance.cdist(x.reshape(1,D),y.reshape(1,D),"cosine")
	elif Distance == 'e':
		D = x.shape[0]
		return distance.cdist(x.reshape(1,D),y.reshape(1,D),"correlation")
	else:
		D = x.shape[0]
		return distance.cdist(x.reshape(1,D),y.reshape(1,D), 'chebyshev')


# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
