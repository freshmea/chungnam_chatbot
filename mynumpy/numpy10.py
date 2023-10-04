# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: hsplit, vsplit
import numpy as np
from numpy03 import npprint


def main():
    a1 = np.arange(48).reshape(6, 2, 4)
    npprint(a1)
    b1, b2, b3 = np.vsplit(a1, 3)
    npprint(b1, b2, b3)

    b4, b5 = np.hsplit(a1, 2)
    npprint(b4, b5)
    b6, b7, b8, b9 = np.dsplit(a1, 4)
    npprint(b6, b7, b8, b9)


if __name__ == "__main__":
    main()
