import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime("199" + x, "%Y-%m")


series = pd.read_csv(
    "pytorch/data/sales.csv",
    header=0,
    parse_dates=[0],
    index_col=0,
    date_parser=parser,
)
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind="kde")
plt.show()
print(residuals.describe())


X = series.values
X = np.nan_to_num(X)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size : len(X)]
history = [x for x in train]
predictinos = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictinos.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f"predicted={yhat}, expected={obs}")
error = mean_squared_error(test, predictinos)
print(f"Test MSE: {error}")
plt.plot(test)
plt.plot(predictinos, color="red")
plt.show()
