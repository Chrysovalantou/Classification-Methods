#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the third assignment in class Methods in Bioinformatics and refers to
Classification alorithms & in particular the K-nn and Naive Bayes.

This particular script is the main call for the functions of:
Knn.py, Bayes.py, Functions.py scripts.

"""

#==============================================================================
#   Libraries:
#==============================================================================

import time
import os
from Functions import*
from collections import Counter

#from Knn import*


print("##..........Classification............##")

# --- Loading Dataset:
Data,Labels = Data_Loading("yeast.data.txt")

# --- Hide samples of the main data set:
Train_set,Hidden_set, Dict_idx = Splitting(Data,Labels)

# --- Splitting the Train Set:

folds = int(input("Give number of folds: "))
Partitions,Hold_indices,Folds = Stratification(Train_set,Dict_idx,folds)

# --- Main methods and Cross validation:
Method = input("Choose Method.\nA: K Nearest Neighbours\nB: Naive Bayes: ")

# --- Mean Accuracy, with k folding training:
Mean_Accuracy = Validation(Partitions,Data,Labels,Hold_indices,Train_set,Folds,Method)

# --- Main Classification: Choose Either method to predict class for hidden set:

print("\n####........... Predicted Classification for new data set...........####")

Hidden_Matrix, Main_Matrix, Main_Labels,Hidden_Labels = Hidden_Values(Data, Train_set, Hidden_set, Labels)

if Method == 'A':
	print("\n####.........Classifier chosen : K nearest neighbours, KNN.........####")
	
	k = int(input("Give number of Neighbours: "))
	Metric = (input("Give Metric Distance:\na. Euclidean\nb. Manhattan\nc. Mahalanobis\nd. Cosine\ne. Correlation\nf. Chebyshev : "))

	Predicted_Classes = Knn(Data,Main_Matrix,Hidden_Matrix,Main_Matrix,Main_Labels,k ,Metric)

else:
	print("\n####...........Classifier chosen : Naive Bayes.....................####")

	Predicted_Classes = Naive_Bayes(Data,Labels,Hold_indices,Train_set,Hidden_set)


print("\nWith mean accuracy {} the predicted class for its sample of the new data set shall be :\n {}".format(Mean_Accuracy,Predicted_Classes))
	
Proportion = Accuracy(Hidden_Labels,Predicted_Classes)

print("\nThe hidden data set has been correctly classified, with {} ".format(round(Proportion*100,3)))
print("Which means that error, subsequently, is : {}".format(100-round(Proportion*100,3)))

c = Counter(Predicted_Classes)

#print("\nMost common class for the samples: {}".format(max(Predicted_Classes,key= Predicted_Classes.count)))
dif_classes = len(set(Predicted_Classes))
#print(dif_classes)
print("\nMost common class for the samples: {}".format(c.most_common(dif_classes)))


# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
