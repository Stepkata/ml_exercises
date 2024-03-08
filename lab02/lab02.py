#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pickle


# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

mnist = fetch_openml('mnist_784', version=1)


# In[7]:


X, y = mnist.data, mnist.target


# In[8]:


y.head(10)


# In[9]:


print((np.array(mnist.data.loc[42]).reshape(28,28)>0).astype(int))


# In[10]:


#3
y = y.astype(int)
y.head(10)


# In[11]:


y = y.sort_values()


# In[12]:


y.head(100)


# In[13]:


X.reindex(index = y.index)


# In[14]:


X_train, X_test=X[:56000], X[56000:]
y_train, y_test=y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[16]:


pd.unique(y_train)


# In[17]:


pd.unique(y_test)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)


# In[19]:


pd.unique(y_train)


# In[20]:


#4
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data,mnist.target.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)



# In[21]:


#4
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
pd.unique(y_train_0)


# In[22]:


#4
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[50]:


#4
pred_train = sgd_clf.predict(X_train)
pred_test = sgd_clf.predict(X_test)
wynik_train = 0
for ind, x in enumerate(y_train_0):
    if pred_train[ind] == x:
        wynik_train+=1
wynik_test=0
for ind, x in enumerate(y_test_0):
    if pred_test[ind] == x:
        wynik_test+=1
w = [wynik_train/len(y_train_0), wynik_test/len(y_test_0)]
with open('sgd_acc.pkl', 'wb') as handle:
    pickle.dump(w, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[51]:


#4
score = cross_val_score(sgd_clf, X_train,y_train_0, cv=3,scoring="accuracy", n_jobs=-1)
with open('sgd_cva.pkl', 'wb') as handle:
    pickle.dump(score, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[46]:


#5
sgd_m_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)


# In[52]:


y_train_pred = cross_val_predict(sgd_clf, X_train,y_train, cv=3, n_jobs=-1)
cm = confusion_matrix(y_train, y_train_pred)
#with open('sgd_cmx.pkl', 'wb') as handle:
    #pickle.dump(cm, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[48]:





# In[ ]:




