# author: choi sugil
# date: 2023.10.04 version: 1.0.0 license: MIT brief: keyward
# description: numpy array definition
import numpy as np


def main():
    li1 = [1, 2, 3, 4, 5]
    a1 = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    a2 = np.array(li1)
    a3 = np.array(range(100))
    print(a1, a2, a3)

    print([1, 2, 3, 4, 5, "string", [1, 2, 3], 0.1])


if __name__ == "__main__":
    main()
