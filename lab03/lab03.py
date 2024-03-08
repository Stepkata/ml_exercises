#!/usr/bin/env python
# coding: utf-8

'''
    Tasks on finding the best fit functions for randomly generated dataset:
      various types of regression.
'''

# In[60]:


import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import sklearn.neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse


# In[61]:


#Generate dataset consisting of random X points and a fourth-degree polynomial function.
size=300
X=np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0=1,2,1,-4,2 #the weights
y=w4*(X**4)+w3*(X**3)+w2*(X**2)+w1*X+w0+np.random.randn(size)*8-4

df=pd.DataFrame({'x': X,'y': y}) #cinvert data to dataframe

# optional: save data to csv for future use
# df.to_csv('dane_do_regresji.csv',index=None)

df.plot.scatter(x='x',y='y')


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
res = [] #table for storing train set & test set accuracy (mse)
X_train =   X_train.reshape(-1, 1)
X_test =    X_test.reshape(-1, 1)


# In[63]:


#linear regression
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
y_pred=lin_reg.predict(X_test)
y_pred_train=lin_reg.predict(X_train)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, '-')
res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])


# In[65]:


#KNN, k=3
knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

knn_3_reg.fit(X_train, y_train)

y_pred=knn_3_reg.predict(X_test)
y_pred_train=knn_3_reg.predict(X_train)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, 'bo')
res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])


# In[66]:

#KNN, k=5
knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)


knn_5_reg.fit(X_train, y_train)

y_pred=knn_5_reg.predict(X_test)
y_pred_train=knn_5_reg.predict(X_train)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, 'bo')

res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])


# In[67]:


#wielomianowa 2, 3, 4, 5 rzedu

poly_feature_2 = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly_feature_2.fit_transform(X_train)
X_poly_test = poly_feature_2.fit_transform(X_test)

poly_2_reg = LinearRegression()

poly_2_reg.fit(X_poly, y_train)

y_pred=poly_2_reg.predict(X_poly_test)
y_pred_train=poly_2_reg.predict(X_poly)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, 'o')

res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])


# In[68]:


poly_feature_3 = PolynomialFeatures(degree=3, include_bias=False)


X_poly = poly_feature_3.fit_transform(X_train)
X_poly_test = poly_feature_3.fit_transform(X_test)

poly_3_reg = LinearRegression()

poly_3_reg.fit(X_poly, y_train)

y_pred=poly_3_reg.predict(X_poly_test)
y_pred_train=poly_3_reg.predict(X_poly)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, 'o')

res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])


# In[69]:


poly_feature_4 = PolynomialFeatures(degree=4, include_bias=False)


X_poly = poly_feature_4.fit_transform(X_train)
X_poly_test = poly_feature_4.fit_transform(X_test)

poly_4_reg = LinearRegression()

poly_4_reg.fit(X_poly, y_train)

y_pred=poly_4_reg.predict(X_poly_test)
y_pred_train=poly_4_reg.predict(X_poly)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, 'o')

res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])


# In[70]:


poly_feature_5 = PolynomialFeatures(degree=5, include_bias=False)

X_poly = poly_feature_5.fit_transform(X_train)
X_poly_test = poly_feature_5.fit_transform(X_test)

poly_5_reg = LinearRegression()

poly_5_reg.fit(X_poly, y_train)

y_pred=poly_5_reg.predict(X_poly_test)
y_pred_train=poly_5_reg.predict(X_poly)

pyplot.plot(X_test, y_test, 'or')
pyplot.plot(X_test, y_pred, 'o')

res.append([mse(y_train, y_pred_train), mse(y_test, y_pred)])
poly_5_reg = poly_5_reg


# In[72]:

#save the results to dataframe
kolumny = ['train_mse', 'test_mse']
wiersze = ['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg']
comparison_res = {wiersze[i]: res[i] for i in range(len(wiersze))}

data = pd.DataFrame.from_dict(comparison_res, orient='index', columns = kolumny)
data.head(7)


# In[77]:

#save the data to pickle
data.to_pickle('varous_regression_mse_comparison.pkl')

