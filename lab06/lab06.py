# %%
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from graphviz import Source
from sklearn.metrics import f1_score
import pickle
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import plot
import sklearn.neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# %%
data_bc = datasets.load_breast_cancer(as_frame=True)

# %%
X=data_bc.data[["mean texture", "mean symmetry"]]
y = data_bc.target
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)



# %%
tree_clf = DecisionTreeClassifier()
#tree_clf.fit(X_train, y_train)
#tree.plot_tree(tree_clf)

# %%
kmeans_clf = sklearn.neighbors.KNeighborsClassifier() 
#kmeans_clf.fit(X_train, y_train)

# %%
log_clf = LogisticRegression()
#log_clf.fit(X_train, y_train)

# %%
voting_clf = VotingClassifier(
    estimators=[('tc', tree_clf),
                ('lc', log_clf),
                ('kc', kmeans_clf)],
    voting='hard')

# %%
voting_clf2 = VotingClassifier(
    estimators=[('tc', tree_clf),
                ('lc', log_clf),
                ('kc', kmeans_clf)],
    voting='soft')

# %%
wyniki = []
for clf in (tree_clf, log_clf, kmeans_clf, voting_clf, voting_clf2):
    clf.fit(X_train, y_train)
    wyniki.append((accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, clf.predict(X_test))))
for item in wyniki:
    print(item)

# %%
with open(r"acc_vote.pkl", "wb") as output_file:
    pickle.dump(wyniki, output_file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open(r"vote.pkl", "wb") as output_file:
    pickle.dump([tree_clf, log_clf, kmeans_clf, voting_clf, voting_clf2],
                output_file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
#2

# %%
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=1.0, bootstrap=True, random_state=42)

# %%
bag50_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=.5, bootstrap=True, random_state=42)

# %%
pas_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=1.0, bootstrap=False, random_state=42)

# %%
pas50_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=.5, bootstrap=False, random_state=42)

# %%
rfor_clf = RandomForestClassifier(n_estimators=30)

# %%
ada_clf = AdaBoostClassifier(n_estimators=30)

# %%
gbrt_clf = GradientBoostingClassifier(n_estimators=30)

# %%
wyniki = []
for clf in (bag_clf, bag50_clf, pas_clf, pas50_clf, rfor_clf, ada_clf, gbrt_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(X_train)
    print(clf.__class__.__name__)
    wyniki.append((accuracy_score(y_train, y_pred2), accuracy_score(y_test, y_pred)))
for item in wyniki:
    print(item)

# %%
with open(r"acc_bag.pkl", "wb") as output_file:
    pickle.dump(wyniki, output_file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open(r"bag.pkl", "wb") as output_file:
    pickle.dump([bag_clf, bag50_clf, pas_clf, pas50_clf, rfor_clf, ada_clf, gbrt_clf],
                output_file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
samp_clf = pas50_clf = BaggingClassifier(
    DecisionTreeClassifier(), max_features=2, n_estimators=30,
    max_samples=.5, bootstrap=True, bootstrap_features=False, random_state=42)

# %%
wyniki = []
samp_clf.fit(X_train, y_train)
wyniki.append((accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, clf.predict(X_test))))

# %%
with open(r"acc_fea.pkl", "wb") as output_file:
    pickle.dump(wyniki, output_file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open(r"fea.pkl", "wb") as output_file:
    pickle.dump([samp_clf], output_file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
for est in samp_clf.estimators_:
    print(est)
    

# %%



