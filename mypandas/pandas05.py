# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: dataframe
import numpy as np
import pandas as pd
from pandas04 import pdprint


def main():
    df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    pdprint(df1)
    df2 = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=np.int8,
        columns=list("가나다"),
    )
    pdprint(df2)

    data = {
        "name": ["Jerry", "Riah", "Paul"],
        "age": [30, 25, 40],
        "dept": ["HR", "SALES", "FINANCE"],
        "children": [2, 3, 4],
    }
    df3 = pd.DataFrame(data)
    pdprint(df3)


if __name__ == "__main__":
    main()
