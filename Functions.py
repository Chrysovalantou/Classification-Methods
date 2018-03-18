#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the third assignment in class Methods in Bioinformatics and refers to
Classification alorithms & in particular the K-nn and Naive Bayes.

This particular script contains functions for:
	 i.splitting the dataset
	ii.cross validation.

"""

#==============================================================================
#   Libraries:
#==============================================================================

import numpy as np
import math
import random

from Knn import*
from Bayes import*


#==============================================================================
#   Data Set loading and split in Hidden & Train:
#==============================================================================

def Data_Loading(filename):

	Data = np.loadtxt(filename,usecols =tuple(np.arange(1,9)))
	Labels = np.loadtxt(filename,usecols=[9] , dtype=str)

	return(Data,Labels)

"""

Splitting tha Data Set:

	Take 10% out of the original data set, for Hidden data. The selection shall be random.
	However, I choose to take 10% of each class. 
	I tried different tricks so as to make sure that hidden will contain or not samples of the smallest class.
	Sometimes it remains, practically, untouchable while some others,the hidden set contains more samples, 
	including samples of the smallest class.
	Thus, I make sure that during Training, each fold will contain samples from each class.

"""
def Splitting(Data,Labels):

	X = Data.T ; N = X.shape[1]  ## X: (8,1484)
	
	real_index = list(range(N))
	
	label_idx = {}				 ## Keys: Classes, Values: the indices of their samples

	for i in range(Labels.shape[0]):
		if Labels[i] not in label_idx.keys():
			label_idx[Labels[i]] = [i] 
		else:
			label_idx[Labels[i]].append(i)
	#print("label_idx = {}".format(label_idx))


	print("\n")
	print("Length of each class:")
	for k in label_idx:
		print("Class {}: {}".format(k,len(label_idx[k])))
	
	Hidden_set = []						## Indices of Hidden set
	Train_set = []						## The rest of them for Train (and Validation) set

	for key in label_idx:
		indices_removed = []
	
		n_01 = math.floor(0.1*len(label_idx[key])) #round(0.1*len(label_idx[key]))	#math.floor(0.1*len(label_idx[key])) 
		n_02 = np.random.choice(np.array(label_idx[key]),n_01,replace = False)
	
		Hidden_set.extend(n_02)
	
		indices_removed.extend(n_02)
		label_idx[key] = list(set(label_idx[key]) - set(indices_removed))  ## remove index of Hidden

	print("\n")
	print("Length of each class after the split:")
	for k in label_idx:
		print("Class {}: {}".format(k,len(label_idx[k])))

	for idx in range(N):
		if idx not in Hidden_set:
			Train_set.append(idx)

	print("Lenght of Train set: {} \nLength of Hidden set: {}".format(len(Train_set),len(Hidden_set)))

	return(Train_set,Hidden_set,label_idx)


#==============================================================================
#   Split the Train Set in specific number of folds:
#==============================================================================


"""
	If given number of folds is greater than the minimum class'es length, I take the length 
	of this class as optimal number of folds.
	Thus, I ensure that its fold will contain not only samples of each class but (at least) 
	one sample of the smallest class.
"""

def Stratification(Train_set,Dict_idx,folds):

	n_03 = min(len(Dict_idx[key]) for key in Dict_idx)
	
	if folds > n_03:
		folds = n_03


	# --- I want practically 5 folds with equal number of indices (samples)
	# --- taken from the Train_set list without replacement
	
	Partitions = []				
								
	for i in range(folds):
		Partitions.append([])

	#fold_size = round(len(Train_set) / folds)
	
	Hold_indices = Dict_idx.copy()

	for k in range(folds):
		for key in Dict_idx:

				indices_removed = [] 
				n_04 = round((1/folds)* len(Hold_indices[key]))
				
				if k <folds -1:
					indices = np.random.choice(np.array(Dict_idx[key]),n_04,replace = False)
				else:
					indices = np.array(Dict_idx[key])

				Partitions[k].extend(indices)
				indices_removed.extend(indices)
				Dict_idx[key] = list(set(Dict_idx[key]) - set(indices_removed))

	for i in range(folds):
		print("Samples in each fold: {} ".format(len(Partitions[i])))

	return(Partitions,Hold_indices,folds)



#==============================================================================
#   Cross Validation with stratified number of folders: 5
#==============================================================================


def Validation(Partitions,Data,Labels,Hold_indices,Train_set,folds,Method):

	print("###........Cross validation with {} folds:........###".format(folds))
 
	
	if Method == 'A':
		print("\n###.........Classifier Chosen : K nearest neighbours, KNN.........###")
		k = int(input("Give number of Neighbours: "))
		Metric = (input("Give Metric Distance:\na. Euclidean\nb. Manhattan\nc. Mahalanobis\nd. Cosine\ne. Correlation\nf. Chebyshev : "))
	else:
		print("\n###.........Classifier Chosen : Naive Bayes.........###")
	
	X = Data.T ; N = X.shape[1]
	Accuracies = []

	for i in range(folds):
		
		Test_idx = Partitions[i]
		Train_idx = list(set(Train_set) - set(Partitions[i]))

		Train_labels = Labels[Train_idx]					## keep current  idx for this folding
		Test_labels  = list(Labels[Test_idx])

		Test_Matrix = np.array(X[:,Test_idx])
		Train_Matrix = np.array(X[:, Train_idx])
		Stack_Matrix = np.array(X[:, Train_set])
		#print(Test_Matrix.shape)
		#print(Train_Matrix.shape)

		if Method == 'A':
			# k = int(input("Give number of Neighbours: "))
			# Metric = (input("Give Metric Distance:\nE: Euclidean\nM: Manhattan\nH: Mahalanobis: "))
			Predictions = Knn(Data,Train_Matrix,Test_Matrix,Stack_Matrix,Train_labels,k,Metric)
		else:
			Predictions = Naive_Bayes(Data,Labels,Hold_indices,Train_idx,Test_idx)

		proportion = Accuracy(Test_labels,Predictions)
		Accuracies.append(proportion)
		
		print("Validation: Fold {}, accuracy: {} ".format(i, round(Accuracies[i]*100,3)))
	
	Mean_Accuracy = round((sum(Accuracies) / float(len(Accuracies)))*100,3)
	Error = 100 - Mean_Accuracy
	print("\nMean Accuracy: {}".format(Mean_Accuracy))
	print("Mean Error: {}".format(Error))

	return(Mean_Accuracy)

def Accuracy(Test_labels,Predictions):
	
	#Accuracies = []
	Common_indices = []
	for label_01, label_02 in zip(Test_labels, Predictions):
		if label_01 == label_02:
			Common_indices.append(label_01)
	#print(len(Common_indices))
	proportion = len(Common_indices)/len(Test_labels)
	#Accuracies.append(proportion)
	
	return(proportion)



def Hidden_Values(Data, Train_set, Hidden_set, Labels):

	X = Data.T ; N = X.shape[1]

	Hidden_Matrix = np.array(X[:,Hidden_set])
	Main_Matrix	  = np.array(X[:,Train_set])

	Hidden_Labels = Labels[Hidden_set]
	Main_Labels	  = Labels[Train_set]

	return(Hidden_Matrix,Main_Matrix,Main_Labels,Hidden_Labels)


# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
