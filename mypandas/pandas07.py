# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: csv
# import numpy as np
import pandas as pd
from pandas04 import pdprint


def main():
    df1 = pd.read_csv("pytorch/data/iris.csv")
    pdprint(df1)

    df_data = df1[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    df_label = df1["Species"]
    pdprint(df_data)
    pdprint(df_label)
    df_data.to_csv("mypandas/iris_data.csv")
    df_label.to_csv("mypandas/iris_label.csv")


if __name__ == "__main__":
    main()
