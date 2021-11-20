#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics Intern Nov Batch 2021
# 
# # Author:Attuluri Rohini
# 
# (GRIPNOV21)
# 
# Data Science & Business Analytics Intern NOV Batch 2021
# 
# TASK 2:Prediction using Unsupervised ML
# 
# From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets
from sklearn.cluster import KMeans


# In[2]:


iris=datasets.load_iris()


# In[3]:


iris


# In[4]:


data = pd.DataFrame(iris.data,columns=iris.feature_names)


# In[5]:


data


# In[6]:


data.isna().any()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.corr()


# In[10]:


X = data.iloc[:,:].values
Y = iris.target


# In[11]:


X


# In[12]:


Y


# In[13]:


#visualising clusters(SEPAL length and width)


# In[14]:


plt.scatter(X[:,0], X[:,1], c=Y, cmap='gist_rainbow_r')
plt.xlabel('SEPAL LENGTH', fontsize=20)
plt.ylabel('SEPAL WIDTH', fontsize=20)


# In[15]:


#petal length and width
plt.scatter(X[:,2], X[:,3], c=Y, cmap='gist_rainbow_r')
plt.xlabel('PETAL LENGTH', fontsize=20)
plt.ylabel('PETAL WIDTH', fontsize=20)


# # elbow method
# #It allows us to pick the optimum amount of clusters for classification

# In[16]:


sos = []
for i in range(1,10):
    km = KMeans(n_clusters = i)
    km.fit(X)
    sos.append(km.inertia_)


# In[17]:


sos


# In[18]:


plt.plot(range(1,10), sos)
plt.title('Elbow Graph')
plt.xlabel('no.of clusters')
plt.ylabel('sos')
plt.show()


# In[19]:


km.inertia_


# # if clusters =3

# In[20]:


Kmeans = KMeans(n_clusters = 3)
Kmeans_Y = Kmeans.fit_predict(X)


# In[21]:


Kmeans.labels_


# In[22]:


#visualising clusters


# In[23]:


plt.scatter(X[Kmeans_Y == 0, 0], X[Kmeans_Y == 0, 1], s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(X[Kmeans_Y == 1, 0], X[Kmeans_Y == 1, 1], s = 100, c = 'red', label = 'Iris-versicolour')
plt.scatter(X[Kmeans_Y == 2, 0], X[Kmeans_Y == 2, 1], s = 100, c = 'violet', label = 'Iris-virginica')
plt.title('SEPAL LENGTH VS SEPAL WIDTH')
#Plotting the centroids of the clusters
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[24]:


plt.scatter(X[Kmeans_Y == 0, 2], X[Kmeans_Y == 0, 3], s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(X[Kmeans_Y == 1, 2], X[Kmeans_Y == 1, 3], s = 100, c = 'red', label = 'Iris-versicolour')
plt.scatter(X[Kmeans_Y == 2, 2], X[Kmeans_Y == 2, 3], s = 100, c = 'violet', label = 'Iris-virginica')
plt.title('SEPAL LENGTH VS SEPAL WIDTH')
#Plotting the centroids of the clusters
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:,1], s = 100, c = 'green', label = 'Centroids')

plt.legend()


# In[ ]:




