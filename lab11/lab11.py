#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
import numpy as np
from scipy.stats import reciprocal
from scikeras.wrappers import KerasRegressor
from scikeras.wrappers import KerasClassifier
import pickle
import keras_tuner as kt
import tensorflow as tf
import os


# In[32]:


housing = fetch_california_housing()


# In[33]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,
                                                              housing.target,
                                                              random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                              y_train_full,
                                                              random_state=42)


# In[34]:


scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.transform(X_test)


# In[35]:


def build_model(n_hidden=1, n_neurons=30, optimizer="sgd", learning_rate=3e-3,
                input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    if optimizer=="sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer=="nesterov":
        optimizer = keras.optimizers.nesterov(learning_rate=learning_rate)
    elif optimizer=="momentum":
        optimizer = keras.optimizers.momentum(learning_rate=learning_rate)
    elif optimizer=="adam":
        optimizer = keras.optimizers.adam(learning_rate=learning_rate)
    else:
        raise Exception('wrong optimizer')
    model.add(keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=optimizer)
    return model


# In[36]:


keras_reg = KerasRegressor(build_model, 
                           callbacks=[keras.callbacks.EarlyStopping(patience=10, 
                                                                    min_delta=1.0, verbose=1)])


# In[37]:


param_distribs = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": np.arange(1, 100),
    "optimizer": ["adam", "sgd", "nesterov"],
    "model__learning_rate": reciprocal(3e-4, 3e-2)
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs,
                                    n_iter=10, cv=3, verbose=2)

rnd_search_cv.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                  verbose=0)


# In[38]:


print(rnd_search_cv.best_params_)


# In[39]:


with open("rnd_search_params.pkl", "wb") as f:
    pickle.dump(rnd_search_cv.best_params_, f)


# In[40]:


try:
    with open("rnd_search_scikeras.pkl", "wb") as f:
        pickle.dump(rnd_search_cv, f)
except Exception:
    print()
    


# In[41]:


def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2,
                                sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam", "nesterov"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer=="adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Nesterov(learning_rate=learning_rate)
    model = tf.keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[8]))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mse", metrics=['mse'], optimizer=optimizer)
    return model 


# In[42]:


random_search_tuner=kt.RandomSearch(build_model_kt, objective="val_mse", max_trials=10,
                                    overwrite=True, directory="my_california_housing",
                                    project_name="my_rnd_search", seed=42)

root_logdir=os.path.join(random_search_tuner.project_dir,'tensorboard')
tb=tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)


# In[43]:


random_search_tuner.search(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[early_stopping_cb, tb])


# In[44]:


best_model = random_search_tuner.get_best_models(num_models=1)[0]
best_hps=random_search_tuner.get_best_hyperparameters(num_trials=1)[0].values
best_hps


with open("kt_search_params.pkl", "wb") as f:
    pickle.dump(best_hps, f)


# In[45]:


try:
    best_model.save("kt_best_model.h5")
except Exception:
    print()


# In[ ]:




