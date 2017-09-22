#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:43:34 2017

@author: sakshi
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd


# Read data from CSV file
df = pd.read_csv('/Users/sakshi/Documents/Books/Machine Learning/dataset_1.csv')
df

# Calculating mean for each variable in the dataset
 temp1 = np.var(df,0)
 temp1

mean_vec = np.mean(df, axis=0)
mean_vec

# Carculating variance for each variable in the dataset
x = df['x']
xmean = sum(x)/len(x)
xmean
xvarRes = sum([(xi - xmean)**2 for xi in x]) / len(x)
xvarRes

y = df['y']
ymean = sum(y)/len(y)
ymean
yvarRes = sum([(xi - ymean)**2 for xi in y]) / len(y)
yvarRes

z = df['z']
zmean = sum(z)/len(z)
zmean
zvarRes = sum([(xi - zmean)**2 for xi in z]) / len(z)
zvarRes

# Calculating covariance between x and y
fig = plt.figure()
ux = fig.add_subplot(1,1,1)
ux.scatter(x,y)
fig.show()
np.cov(x,y)

# Calculating covariance between y and z
fig = plt.figure()
ux = fig.add_subplot(1,1,1)
ux.scatter(y,z)
fig.show()
np.cov(y,z)

# Performing PCA on the dataset
cov=np.cov(df.T)
cov

eigen_val ,eigen_vec = np.linalg.eig(cov)
eigen_vec
eigen_val

eigen_pair = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]
eigen_pair

eigen_pair.sort()

eigen_pair.reverse()

eigen_pair


W = np.hstack((eigen_pair[0][1].reshape(3,1), eigen_pair[1][1].reshape(3,1), eigen_pair[0][1].reshape(3,1)))
W
Y = df.dot(W)
Y


# Solution for question 3 part B
a = np.array([[0,-1],[2,3]], dtype=int)

eigen_val_a ,eigen_vec_a = np.linalg.eig(a)
eigen_val_a
eigen_vec_a