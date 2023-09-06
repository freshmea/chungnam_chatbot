import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('pytorch/data/weather.csv')

# 그래프 그리기
# dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plt.title('MinTemp vs MaxTemp')
# plt.xlabel('MinTemp')
# plt.ylabel('MaxTemp')
# plt.show()

X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# 선형 회귀로는 비교 안됨. 
# print(sklearn.metrics.accuracy_score(y_test, y_pred))

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

#스캐터 그래프
# plt.scatter(X_test, y_test, color='blue')
# plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()

print('평균제곱 에러: ', sklearn.metrics.mean_squared_error(y_test, y_pred))
print('루트 평균제곱 에러: ', np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))
