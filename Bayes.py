#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Chrysovalantou Kalaitzidou

The current code is part of the third assignment in class Methods in Bioinformatics and refers to
Classification alorithms & in particular the K-nn and Naive Bayes.

This particular script contains the implementation of Naive Bayes
classifier.

"""

from Functions import*
import scipy.stats



def Naive_Bayes(Data,Labels,Hold_indices,Train_set,Test_set):

	#Train_set,Test_set,Data_dict = Splitting(Data,Labels)

	X = Data.T ; D= X.shape[0]; N = X.shape[1]

	real_index = list(range(N))
	k = len(Hold_indices) #; print(k)
	
	Order_labels = []			## Ordering Labels 
	
	for i in Hold_indices.keys():
		Order_labels.append(i)
		
	Class_Priors = np.zeros(k)
	
	for c in range(len(Order_labels)):
		key = Order_labels[c]
		Class_Priors[c] = len(Hold_indices[key]) / len(Train_set)
		
	print("The Prior Probabilty of its Class is: {}".format(Class_Priors))
	
	### Classes: 10 objects, each one an array of the samples belonging to this particular class.
	##	Means:	 10 objects, each one an array of the means of each variable, in this particular class.
	#	Stdv:	 10 objects, each one an array of the std of each variable, in this particular class.
	
	Classes = np.zeros(k,dtype = object)
	Means = np.zeros(k,dtype = object)
	Stdv  = np.zeros(k,dtype = object)

	for c in range(len(Order_labels)):
		key = Order_labels[c]
		
		Classes[c] = np.array(X[:,Hold_indices[key]])
		Means[c]  = np.mean(Classes[c], axis =1)
		Stdv[c]	  = np.std(Classes[c], axis=1)
		
	Train_Array = np.array(X[:, Train_set]); print("Shape of Train Matrix: {}".format(Train_Array.shape))
	Test_Array   = np.array(X[:, Test_set]);  print("Shape of Test Matrix: {}".format(Test_Array.shape))
	
	m = Test_Array.shape[1]		#  Samples
	n = Test_Array.shape[0]		#  Variables 

	Predicted_Labels = []

	for i in range(m):
		Probability = np.zeros(k, dtype=object)
		for c in range(len(Order_labels)):
			key = Order_labels[c]
			Predictions = np.zeros(n)
			for j in range(n):
				Conditional = scipy.stats.norm.pdf(Test_Array[j,i], loc = Means[c][j], scale = Stdv[c][j])
				## zero conditional probability
				if Conditional == 0 or np.isnan(Conditional):
					#print(Test_Array[j,i])
					vec = np.array(X[j, Hold_indices[key]])
					n_c = list(vec).count(Test_Array[j,i]) #; print("nc:{}".format(n_c))
					n_l = len(Hold_indices[key])
					t 	= len(set(Hold_indices[key])); p = 1/t
					m 	= 1
					
					Conditional = (n_c + m*p)/(n_l + m)
				
				Predictions[j] = Conditional
			Product = np.prod(Predictions)
			Probability[c] = np.dot(Product,Class_Priors[c])

		Max_Prob = np.argmax(Probability)
		Predicted_Labels.append(Order_labels[Max_Prob])
	
	return(Predicted_Labels)
	
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
# ------  ------  ------  ------  ------ ------  ------  ------  -------  ----
