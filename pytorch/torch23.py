import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


def parser(x):
    return datetime.strptime("199" + x, "&Y-&m")


series = pd.read_csv("pytorch/data/sales.csv")
