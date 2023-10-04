# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: multi dimensional array slicing, indexing
import numpy as np
from numpy03 import npprint


# def func1(x, y, z):
#     return x + y * 2 + z


def main():
    a1 = np.fromfunction(lambda x, y, z: x + y + z, (2, 5, 4), dtype=np.int8)
    npprint(a1)
    a2 = a1[:, 1::2, :3]
    npprint(a2)
    a2[1, ...] = -1
    npprint(a2)


if __name__ == "__main__":
    main()
