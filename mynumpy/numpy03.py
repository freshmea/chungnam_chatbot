import numpy as np


def npprint(*arrays: np.ndarray) -> None:
    for array in arrays:
        print(
            f"Data Type: {array.dtype}, Data dim: {array.ndim}, Data Shape: {array.shape}"
        )
        print("-" * 20)
        print(array)


def main():
    a1 = np.zeros((50, 50), dtype=np.int8)
    a2 = np.ones((30, 30), dtype=np.int8)
    a3 = np.full((50, 50), 4.5, dtype=np.float32)

    a11 = np.zeros_like(a2)
    a22 = np.ones_like(a1)
    a33 = np.full_like(a2, 33)

    a4 = np.eye(5, k=0, dtype=np.int8)
    a5 = np.linspace(1, 25, 25).reshape((5, 5))
    a6 = np.diag(a5, k=0)
    npprint(a1)
    npprint(a2)
    npprint(a3)
    npprint(a4)
    npprint(a5)
    npprint(a6)


if __name__ == "__main__":
    main()
