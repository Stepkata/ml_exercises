#!/usr/bin/env python
# coding: utf-8
"""Using decision trees for classification and finding best fit 
functions for randomised dataset

"""
# In[19]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from graphviz import Source
from sklearn.metrics import f1_score
import pickle
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import plot
from sklearn.metrics import accuracy_score


# In[20]:


data_breast_cancer = datasets.load_breast_cancer(as_frame = True)
#print(data_breast_cancer["DESCR"])


# In[21]:


X=data_breast_cancer.data[["mean texture", "mean symmetry"]]
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, shuffle=True)


# In[22]:


deph = 3
tree_clf = DecisionTreeClassifier(max_depth=deph, random_state=42)
tree_clf.fit(X_train, y_train)
tree.plot_tree(tree_clf)
tree.export_graphviz(tree_clf, out_file='bc.png', feature_names=X.columns)


# In[23]:


y_pred = tree_clf.predict(X_train)
f_train = f1_score(y_train, y_pred, average='macro')
print(f_train)
y_pred2 = tree_clf.predict(X_test)
f_test = f1_score(y_test, y_pred2, average='macro')
print(f_test)


# In[24]:


score_train = accuracy_score(y_train, y_pred)
print(score_train)
score_test = accuracy_score(y_test, y_pred2)
print(score_test)


# In[25]:


result = [deph, f_train, f_test, score_train, score_test]
with open(r'f1acc_tree.pkl', "wb") as output_file:
    pickle.dump([deph, f_train, f_test, score_train, score_test], output_file, protocol=pickle.HIGHEST_PROTOCOL)


# In[26]:


size = 300
X = np.random.rand(size)*5-2.5
w4,w3,w2,w1,w0 = 1,2,1,-4,2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x':X, 'y':y })
df.plot.scatter(x='x', y='y')


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[28]:


depth =3
tree_reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
tree_reg.fit(X_train, y_train)
tree.plot_tree(tree_clf)
tree.export_graphviz(tree_reg, out_file='reg.png')


# In[29]:


y_pred = tree_reg.predict(X_train)
mse_train = mean_squared_error(y_train,y_pred)
print(mse_train)
y_pred = tree_reg.predict(X_test)
mse_test = mean_squared_error(y_test,y_pred)
print(mse_test)


# In[30]:


plot(X_test, y_test, 'bo')
plot(X_test, y_pred, 'ro')


# In[31]:


result = [depth, mse_train, mse_test]
with open(r"mse_tree.pkl", "wb") as output_file:
    pickle.dump([depth, mse_train, mse_test], output_file, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:




