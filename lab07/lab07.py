#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pickle
from sklearn.metrics import confusion_matrix


# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto') 
mnist.target= mnist.target.astype(np.uint8) 
X = mnist["data"] 
y = mnist["target"]


# In[3]:


kmeans = []
blobs = [8,9,10,11,12]
wyniki = []
for k in blobs:
    km = KMeans(n_clusters = k, random_state=42).fit(X)
    wyniki.append(silhouette_score(X, km.labels_))
    kmeans.append(km)

for item in wyniki:
    print(item)


# In[4]:


with open(r"kmeans_sil.pkl", "wb") as output_file:
     pickle.dump(wyniki, output_file)


# In[5]:


cmatrix = confusion_matrix(y, kmeans[2].predict(X))
cmatrix


# In[6]:


indexes = []
for row in cmatrix:
    indexes.append(np.argmax(row))
print(indexes)
indexes = list(set(sorted(indexes)))
print(indexes)


# In[7]:


with open(r"kmeans_argmax.pkl", "wb") as output_file:
     pickle.dump(indexes, output_file)


# In[26]:


XX = X[:300]
xodl = [np.inf for x in range(300)]


# In[27]:


for index, xx in enumerate(XX):
    for x in X[300:]:
        xodl[index] = min(xodl[index], np.linalg.norm(x-xx))

xodl.sort()
filter(lambda a: a != 0, xodl)
wyniki = xodl[:10]
print(wyniki)


# In[28]:


with open(r"dist.pkl", "wb") as output_file:
     pickle.dump(wyniki, output_file)


# In[31]:


s = np.mean(wyniki[:3])
epss = np.arange(s, s+0.1*s, 0.04*s)
dbscans = []
for eps in epss:
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)
    #print("done!")
    dbscans.append(dbscan)


# In[32]:


wyniki_dbscan = []
for dbscan in dbscans:
    wyniki_dbscan.append(len(list(set(dbscan.labels_))))
print(wyniki_dbscan)
with open(r"dbscan_len.pkl", "wb") as output_file:
     pickle.dump(wyniki_dbscan, output_file)


# In[ ]:





# In[ ]:




