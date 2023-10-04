# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: npprint user-definition print function, zeros, ones, full, eye, diag
import numpy as np


def npprint(*arrays: np.ndarray) -> None:
    for array in arrays:
        print(f"Data Type: {array.dtype}, ", end="")
        print(f"Data dim: {array.ndim}, ", end="")
        print(f"Data Shape: {array.shape}")
        print("-" * 20)
        print(array)


def main():
    a1 = np.zeros((50, 50), dtype=np.int8)
    a2 = np.ones((30, 30), dtype=np.int8)
    a3 = np.full((50, 50), 4.5, dtype=np.float32)

    a11 = np.zeros_like(a2)
    a22 = np.ones_like(a1)
    a33 = np.full_like(a2, 33)

    a4 = np.eye(5, k=0, dtype=np.int8)
    a5 = np.linspace(1, 25, 25).reshape((5, 5))
    a6 = np.diag(a5, k=0)
    npprint(a1, a11, a2, a22, a3, a33, a4, a5, a6)


if __name__ == "__main__":
    main()
