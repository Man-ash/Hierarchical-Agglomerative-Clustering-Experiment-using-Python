#!/usr/bin/env python
# coding: utf-8

# # Implementing Hierarchical Clustering 

# In[1]:


#Importing required libraries


import matplotlib.pyplot as plt  
import pandas as pd  
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np  


# In[5]:


#Loading the dataset


customer_data = pd.read_csv('shopping_data.csv')  
customer_data.shape
customer_data.head()


# In[6]:


data = customer_data.iloc[:, 3:].values
data.shape


# In[7]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  
plt.title("Dendogram")  
dend = shc.dendrogram(shc.linkage(data,method='ward'))


# # Agglomentric Clustering

# In[8]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data) 


# In[10]:


#plot the clusters to see how actually our data has been clustered


plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()


# In[11]:


## MANASH PRATIM KAKATI
## PG CERTIFICATION IN AI & ML
## E&ICT ACADAMY, IIT GUWAHATI 


# In[ ]:




