# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: indexing
import numpy as np
import pandas as pd


def pdprint(*pandas_datas: pd.DataFrame | pd.Series) -> None:
    for data in pandas_datas:
        print(f"Data Type: {data.dtypes}, ", end="")
        print(f"Data dim: {data.ndim}, ", end="")
        print(f"Data Shape: {data.shape}")
        print("-" * 20)
        print(data)


def main():
    a = np.array([0])
    print(a)
    s1 = pd.Series(["마케팅", "경영", "개발", "기획", "전략"], index=list("abcde"), name="부서명")
    print(s1.shape)
    print(s1.name)
    print(s1.size)
    print(s1.ndim)
    print(s1.values)
    print(s1.dtypes)
    print(s1.isna())


if __name__ == "__main__":
    main()
