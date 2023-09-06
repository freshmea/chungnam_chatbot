import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import sklearn.metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


digits = load_digits()
# 그래프로 나타내기
# print(digits.data.shape)
# print(digits.target.shape)
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
#     plt.subplot(1, 5, index+1)
#     plt.imshow(np.reshape(image, (8,8)) )
#     plt.title("Training: %i\n" % label, fontsize= 20)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size= 0.25, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

print(logisticRegr.predict(X_test[0].reshape(1,-1)))
print(logisticRegr.predict(X_test[0:10]))

# metric  을 이용해서 결과 비교
prediction = logisticRegr.predict(X_test)
print(sklearn.metrics.accuracy_score(y_test, prediction))

# 내부 score 메소드를 이용해서 결과 비교
score = logisticRegr.score(X_test, y_test)
print(score)


cm = sklearn.metrics.confusion_matrix(y_test, prediction)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap= 'Blues_r')
plt.ylabel('Actual lable')
plt.xlabel('Predicted label')
all_sample_title = f'Accuracy Score:{score}'
plt.title(all_sample_title, size = 15)
plt.show()
