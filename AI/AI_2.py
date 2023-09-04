import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'chap02/data/car_evaluation.csv')
print(dataset.iloc(0))
print(dataset.head())
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind='pie', autopct='%0.05f%%', colors=['lightblue', 'lightgreen', 'orange', 'pink'], explode=(0.05, 0.05, 0.05,0.05))
plt.show()
