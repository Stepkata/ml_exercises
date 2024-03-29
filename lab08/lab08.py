#!/usr/bin/env python
# coding: utf-8

"""
    Principle Component Analysis exercise
"""

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

# %%
def play_with_pca(data):
    pca = PCA(n_components=0.9)
    scaler = StandardScaler()
    reduced = pca.fit_transform(scaler.fit_transform(data.data))
    print(pca.explained_variance_ratio_)

    wyniki = []
    for i in range(len(pca.components_)):
        c0_important_feature = np.argmax(abs(pca.components_[i]))
        wyniki.append(c0_important_feature)
    print(wyniki)

# In[57]:
play_with_pca(data_bc)


# In[ ]:
play_with_pca(data_ir)



