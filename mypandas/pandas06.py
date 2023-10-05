# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: default
# import numpy as np
import pandas as pd
from pandas04 import pdprint


def main():
    excel = pd.read_excel(
        "mypandas/test1.xlsx", sheet_name="Sheet1", engine="openpyxl", header=None
    )
    pdprint(excel)

    excel.to_excel("mypandas/test2.xlsx", sheet_name="Sheet1", header=None, index=False)


if __name__ == "__main__":
    main()
