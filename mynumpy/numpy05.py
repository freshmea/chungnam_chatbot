# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: basic operation
import numpy as np
from numpy03 import npprint


def main():
    a1 = np.array([0, 30, 60, 90, 0, 0])
    a2 = np.linspace(5, 6, a1.size).reshape(a1.shape)
    b1 = a1 - a2
    # b2 = b1**3
    b2 = np.power(b1, 3)
    b3 = np.sin(a1 / 180 * np.pi)
    b4 = a1 % a2
    b5 = b1 < 25

    npprint(b1, b2, b3, b4, b5)
    print(locals()["a1"])


if __name__ == "__main__":
    main()
