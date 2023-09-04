import torch1
from torch1 import Torch

class PyTorch(Torch):
    def __init__(self, name):
        super().__init__(name)
        self.pyname = 'py최수길'

    def sub_print(self):
        print('파이네임 : ' + self.pyname+ '\n이름은 : ' + self.name)


def main():
    # t = torch1.Torch('최수길')
    t = Torch('최수길')
    t2 = PyTorch('최아무개')
    t.print()
    # t.subprint() # 부모 클래스에 없음.
    t2.print()
    t2.sub_print()

if __name__ == '__main__':
    main()