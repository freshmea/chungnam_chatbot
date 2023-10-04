# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: vstack, hstack, concatenate
import numpy as np
from numpy03 import npprint


def main():
    a1 = np.fromfunction(lambda x, y, z: x + y + z, (2, 5, 4), dtype=np.int8)

    a2 = np.arange(1, a1.size // 2 + 1).reshape((1, 5, 4)) * 10
    npprint(a2)
    # a3 = np.vstack((a1[0], a2[0]))
    a3 = np.vstack((a1[0, ...], a2[0, ...]))
    a4 = np.hstack((a1[1, ...], a2[0, ...]))
    npprint(a3)
    npprint(a4)
    a2 = np.concatenate((a2, a2), axis=0)
    a5 = np.concatenate((a1, a2), axis=2)
    npprint(a5)


if __name__ == "__main__":
    main()
