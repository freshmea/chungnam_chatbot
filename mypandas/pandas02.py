# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: indexing, fancy indexing
import numpy as np
import pandas as pd
import time


def main():
    pd1 = pd.Series(["부장", "차장", "과장", "대리"], dtype="string")
    print(pd1[2])
    print(pd1[[0, 2, 3]])

    np.random.seed(int(time.time()))
    pd2 = pd.Series(np.random.randint(0, 20000, size=(50,)))
    print(pd2)
    filter = pd2 > 15000
    print(pd2[filter])


if __name__ == "__main__":
    main()
