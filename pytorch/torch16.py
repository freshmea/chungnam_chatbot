import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

X = pd.read_csv('pytorch/data/credit card.csv')

X.drop('CUST_ID', axis=1, inplace=True)
X.fillna(method='ffill', inplace=True)
print(X.head())

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
print(X_scaled)
X_normalized = normalize(X_scaled)
# X_normalized = pd.DataFrame(X_normalized) # 데이타프레임타입의 노말라이즈데이터!
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
print(type(X_principal))
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())

# db_default = DBSCAN(eps= 0.0375, min_samples=3).fit(X_principal)
# labels = db_default.labels_

# colours = {0:'r', 1:'g', 2:'b', -1:'k'}
# cvec = [colours[label] for label in labels]
# r = plt.scatter(X_principal['P1'], X_principal['P2'], color = 'r')
# g = plt.scatter(X_principal['P1'], X_principal['P2'], color = 'g')
# b = plt.scatter(X_principal['P1'], X_principal['P2'], color = 'b')
# k = plt.scatter(X_principal['P1'], X_principal['P2'], color = 'k')

# plt.figure(figsize=(9,9))
# plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)

# plt.legend((r, g, b, k), ('label 0', 'label 1', 'label 2', 'label 3'))
# plt.show()

db_default = DBSCAN(eps= 0.0375, min_samples=50).fit(X_principal)
labels = db_default.labels_

colours = {0:'r', 1:'g', 2:'b', 3:'c', 4:'y', 5:'m', -1:'k'}
cvec = [colours[label] for label in labels]
# colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
colors1 =list('rgbcymk')
scatter_li = []
for c in colors1:
    scatter_li.append(plt.scatter(X_principal['P1'], X_principal['P2'], marker = 'o' ,color = c))

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)

plt.legend(
    (tuple([sc for sc in scatter_li])),
    ('label 0', 'label 1', 'label 2', 'label 3','label 4', 'label 5', 'label -1'),
    scatterpoints = 1,
    loc = 'upper left',
    ncol = 3,
    fontsize = 8)
plt.show()