"""
    KMeans and DBScan studies
"""

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto') 
mnist.target= mnist.target.astype(np.uint8) 
X = mnist["data"] 
y = mnist["target"]

# %%
def visualise(x: list, y: list, filename: str):

    plt.bar(x, y, color='blue')
    plt.title('Accuracy')
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy Scores')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically

    # Save the plot to a PNG file
    plt.savefig(filename, bbox_inches='tight')  # Adjust the bounding box to include rotated labels

    # Display the plot
    plt.show()

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

visualise(blobs, wyniki, "kmeans_sil_score")

# In[5]:

#example confusion matrix
cmatrix = confusion_matrix(y, kmeans[2].predict(X))
print(cmatrix)


# In[26]:

#prepare data for dbscan
XX = X[:300]
xodl = [np.inf for x in range(300)]


# In[27]:


for index, xx in enumerate(XX):
    for x in X[300:]:
        xodl[index] = min(xodl[index], np.linalg.norm(x-xx))

xodl.sort()
filter(lambda a: a != 0, xodl)
values = xodl[:10]
print(values)


# In[31]:


s = np.mean(values[:3])
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
visualise(epss, wyniki_dbscan, "dbscan")




