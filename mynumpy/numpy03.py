import numpy as np


def main():
    a1 = np.zeros((50, 50), dtype=np.int8)
    a2 = np.ones((50, 50), dtype=np.int8)
    a3 = np.full((50, 50), 4.5, dtype=np.float32)
    a4 = np.eye(5, k=0, dtype=np.int8)
    a5 = np.linspace(1, 25, 25).reshape((5, 5))
    a6 = np.diag(a5, k=0)
    print(a1, a1.dtype, a1.size, a1.ndim, a1.shape)
    print(a2, a2.dtype, a2.size, a2.ndim, a2.shape)
    print(a3, a3.dtype, a3.size, a3.ndim, a3.shape)
    print(a4, a4.dtype, a4.size, a4.ndim, a4.shape)
    print(a5, a5.dtype, a5.size, a5.ndim, a5.shape)
    print(a6, a6.dtype, a6.size, a6.ndim, a6.shape)


if __name__ == "__main__":
    main()
