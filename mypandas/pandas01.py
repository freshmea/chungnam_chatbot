# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: Series
import numpy as np
import pandas as pd


def main():
    a1 = np.arange(100, 105)
    pd1 = pd.Series(a1)
    print(pd1)
    pd2 = pd.Series(a1, dtype="int8")
    print(pd2)
    pd3 = pd.Series(["부장", "차장", "과장", "대리"], dtype="string")
    print(pd3)


if __name__ == "__main__":
    main()
