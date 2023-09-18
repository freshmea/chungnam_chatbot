# author: choi sugil
# date: 2019.12.02 version: 1.0.0 license: MIT brief: keyward

import keyword


def main():
    print(keyword.kwlist)
    for i in keyword.kwlist:
        print(i)
    print(len(keyword.kwlist))


if __name__ == "__main__":
    main()
