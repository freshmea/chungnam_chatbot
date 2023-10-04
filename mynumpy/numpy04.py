# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: reshape

import numpy as np
from numpy03 import npprint


def main():
    # a2 = np.full((6,3), -1).reshape((3,3,2))
    a1 = np.full((6, 3), -1)
    a2 = np.reshape(a1, (3, 3, 2))
    a3 = np.arange(1, 10 + 1, 0.5).reshape((4, 5))
    try:
        a4 = np.reshape(a3, (5, 5))
        npprint(a4)
    except Exception:
        print("Error")
    a5 = np.linspace(1, 10 + 1, 20).reshape((2, 5, 2))
    npprint(a1, a2, a3, a5)


if __name__ == "__main__":
    main()
