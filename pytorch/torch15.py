import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

os.environ['OMP_NUM_THREADS'] = '2'

data = pd.read_csv('pytorch/data/sales data.csv')
print(data.head())

categorical_features = ['Channel', 'Region']
continuous_features = ['Fresh',  'Milk',  'Grocery',  'Frozen',  'Detergents_Paper',  'Delicassen']

for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
print(data.head())

mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)
print(data_transformed)

sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    print(km.labels_)
    sum_of_squared_distances.append(km.inertia_)

# print(sum_of_squared_distances)
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('Optimal K')
plt.show()