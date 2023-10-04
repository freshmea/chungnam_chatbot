# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: arange, linspace
import numpy as np


def main():
    a1 = np.arange(20)
    a2 = np.arange(5, 15)
    a3 = np.arange(0, 1, 0.1)
    a4 = np.linspace(1, 10, 12)

    print(a1, a2, a3)

    print(a4)


if __name__ == "__main__":
    main()
