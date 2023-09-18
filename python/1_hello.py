# 연속으로 hello, python을 출력하는 코드
# input으로 받은 숫자만큼 반복하여 출력한다.
# 2023.09.19
# author: choi sugil
# license: MIT
def main():
    input_num = int(input("반복할 횟수를 입력하세요: "))
    for _ in range(input_num):
        print("Hello, python!")


if __name__ == "__main__":
    main()
