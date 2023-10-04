# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: broadcasting
import numpy as np
from numpy03 import npprint


def main():
    a1 = np.linspace(1, 12, 12).reshape((3, 4))
    a2 = np.arange(1, 5)  # element number = 4
    a3 = a2.reshape((4, 1))  # element number = 4
    a4 = a1.reshape(3, 1, 4)

    # broadcasting
    b1 = a1 + a2
    b2 = a2 + a3
    b3 = a1 + a4

    npprint(a1, a2, a3, a4)
    npprint(b1, b2, b3)


if __name__ == "__main__":
    main()
