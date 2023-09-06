import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import pickle
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('pytorch/data/iris.data', names=names)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(type(X_test))
from sklearn.preprocessing import StandardScaler
s = StandardScaler()

X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)
print(X_train)
print(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)

with open('pytorch/data/knn.picke', 'bw') as f:
    pickle.dump(knn, f)
    
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("정확도: {}".format( accuracy_score(y_test, y_pred)))
print(y_pred)

k=10
acc_array=np.zeros(k)
for k in np.arange(1,k+1,1): 
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train) 
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1]=acc 

max_acc=np.amax(acc_array)
acc_list=list(acc_array)
k=acc_list.index(max_acc)
print("정확도 ", max_acc, "으로 최적의 k는", k+1, "입니다.")

while True:
    a = float(input('input number '))
    b = float(input('input number '))
    c = float(input('input number '))
    d = float(input('input number '))
    X_test = np.array([[a,b,c,d]])
    X_test = s.fit_transform(X_test)
    y_pred = knn.predict(X_test)
    print(y_pred)