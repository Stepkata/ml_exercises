#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn import datasets
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler


# In[10]:


data_bc = load_breast_cancer()
data_ir = load_iris()


# In[57]:


pca = PCA(n_components=0.9)
scaler = StandardScaler()
reduced_bc = pca.fit_transform(scaler.fit_transform(data_bc.data))
print(pca.explained_variance_ratio_)


# In[49]:


with open('pca_bc.pkl', 'wb') as handle:
    pickle.dump(pca.explained_variance_ratio_, handle)


# In[58]:


wyniki = []
for i in range(len(pca.components_)):
    c0_important_feature = np.argmax(abs(pca.components_[i]))
    wyniki.append(c0_important_feature)
print(wyniki)


# In[65]:


with open('idx_bc.pkl', 'wb') as handle:
    pickle.dump(wyniki, handle)


# In[66]:


pca = PCA(n_components=0.9)
scaler = StandardScaler()
reduced_ir = pca.fit_transform(scaler.fit_transform(data_ir.data))
print(pca.explained_variance_ratio_)


# In[67]:


with open('pca_ir.pkl', 'wb') as handle:
    pickle.dump(pca.explained_variance_ratio_, handle)


# In[70]:


wyniki = []
for i in range(len(pca.components_)):
    c0_important_feature = np.argmax(abs(pca.components_[i]))
    wyniki.append(c0_important_feature)
print(wyniki)


# In[69]:


with open('idx_ir.pkl', 'wb') as handle:
    pickle.dump(wyniki, handle)


# In[ ]:




