# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: default
import numpy as np
import pandas as pd
from pandas04 import pdprint


def main():
    a = np.array([0])
    b = pd.Series(a)
    print(a, b)


if __name__ == "__main__":
    main()
