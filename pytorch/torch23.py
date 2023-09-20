import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


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
