#!/usr/bin/env python
# coding: utf-8

'''
    Using LinearSVC and StandardScaler for simple 
    classification execises
'''
# In[42]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle


# In[3]:


data_breast_cancer = datasets.load_breast_cancer()
print(data_breast_cancer["DESCR"])


# In[4]:


data_iris = datasets.load_iris()
print(data_iris["DESCR"])


# In[18]:


X, y = data_breast_cancer["data"][:, (3,4)], data_breast_cancer.target

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[19]:


X, y = data_iris["data"][:, (2,3)], data_iris.target

Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[33]:


svm_clf = LinearSVC(C=1,loss="hinge",random_state=42)
svm_clf.fit(Xc_train, yc_train)


# In[34]:


svm_clf2 = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1,
                             loss="hinge",
                             random_state=42)),
])
svm_clf2.fit(Xc_train, yc_train)


# In[36]:


pred = svm_clf.predict(Xc_train)
pred_test = svm_clf.predict(Xc_test)
pred2 = svm_clf2.predict(Xc_train)
pred_test2 = svm_clf2.predict(Xc_test)


# In[41]:


result1 = accuracy_score(yc_train, pred, normalize=True)
result2 = accuracy_score(yc_test, pred_test, normalize=True)
result3 = accuracy_score(yc_train, pred2, normalize=True)
result4 = accuracy_score(yc_test, pred_test2, normalize=True)

print(result1, result2, result3, result4)


# In[45]:


svm_clf = LinearSVC(C=1,loss="hinge",random_state=42)
svm_clf.fit(Xi_train, yi_train)


# In[46]:


svm_clf2 = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1,
                             loss="hinge",
                             random_state=42)),
])
svm_clf2.fit(Xi_train, yi_train)


# In[47]:


pred = svm_clf.predict(Xi_train)
pred_test = svm_clf.predict(Xi_test)
pred2 = svm_clf2.predict(Xi_train)
pred_test2 = svm_clf2.predict(Xi_test)


# In[48]:


result1 = accuracy_score(yi_train, pred, normalize=True)
result2 = accuracy_score(yi_test, pred_test, normalize=True)
result3 = accuracy_score(yi_train, pred2, normalize=True)
result4 = accuracy_score(yi_test, pred_test2, normalize=True)

print(result1, result2, result3, result4)

