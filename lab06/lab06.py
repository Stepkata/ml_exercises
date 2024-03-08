"""
    Comparison of various simple classifiers
"""

# %%
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import sklearn.neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
# %%
data_bc = datasets.load_breast_cancer(as_frame=True)

# %%
X=data_bc.data[["mean texture", "mean symmetry"]]
y = data_bc.target
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)

# %%
def train_test_classifiers(classifier_list: list, filename: str = "combined_accuracy_plots.png") -> None:
    wyniki_train: list = []
    wyniki_test: list = []
    classifiers: list = []
    for i, clf in enumerate(classifier_list):
        clf.fit(X_train, y_train)
        wyniki_train.append(accuracy_score(y_train, clf.predict(X_train)))
        wyniki_test.append(accuracy_score(y_test, clf.predict(X_test)))
        classifiers.append(str(i)+". " + clf.__class__.__name__)
        
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    # Plot for train dataset accuracy
    ax1.bar(classifiers, wyniki_train, color='blue')
    ax1.set_title('Train Dataset Accuracy')
    ax1.set_xlabel('Classifiers')
    ax1.set_ylabel('Accuracy Scores')
    ax1.tick_params(axis='x', rotation=90)

    # Plot for test dataset accuracy
    ax2.bar(classifiers, wyniki_test, color='blue')
    ax2.set_title('Test Dataset Accuracy')
    ax2.set_xlabel('Classifiers')
    ax2.set_ylabel('Accuracy Scores')
    ax2.tick_params(axis='x', rotation=90)

    plt.tight_layout()

    # Save the combined plot to a PNG file
    plt.savefig(filename)
    plt.show()

# %%
tree_clf = DecisionTreeClassifier()
# %%
kmeans_clf = sklearn.neighbors.KNeighborsClassifier() 

# %%
log_clf = LogisticRegression()

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
samp_clf = pas50_clf = BaggingClassifier(
    DecisionTreeClassifier(), max_features=2, n_estimators=30,
    max_samples=.5, bootstrap=True, bootstrap_features=False, random_state=42)

# %%

train_test_classifiers([tree_clf, log_clf, kmeans_clf, voting_clf, voting_clf2, bag_clf, bag50_clf, pas_clf, pas50_clf, rfor_clf, ada_clf, gbrt_clf, samp_clf])

# %%
