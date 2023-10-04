# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: indexing, slicing
import numpy as np
from numpy03 import npprint


def main():
    a1 = np.arange(1, 10 + 1) ** 2
    a2 = a1[2:9]  # reference not copy
    npprint(a1, a2)
    a1[3] = a1[1] + a1[2]
    npprint(a1, a2)
    # a2[0:5:2]
    a2[:5:2] = 10_000

    # pythonic code
    # for a in a1[::-1]:
    #     print(a, end=", ")

    for i in range(len(a1)):
        print(a1[(i + 1) * -1], end=", ")
    print()


if __name__ == "__main__":
    main()
