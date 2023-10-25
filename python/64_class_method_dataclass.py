# author: choi sugil
# date: 2023.09.19 version: 1.0.0 license: MIT brief: keyward
# description: 데이터클래스 와 메서드를 사용하는 프로그램
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Student:
    name: str
    korean: int
    math: int
    english: int
    science: int

    def get_sum(self):
        return self.korean + self.math + self.english + self.science

    def get_average(self):
        return self.get_sum() / 4

    def to_string(self):
        return f"{self.name}\t{self.get_sum()}\t{self.get_average():.2f}"


def main():
    students = []
    param_dir = r"C:\chungnam_chatbot\python"
    with open(Path(param_dir) / "params.json", "r", encoding="UTF8") as f:
        params = json.loads(f.read())
    for param in params:
        students.append(Student(**param))
    print("이름\t총점\t평균")
    for student in students:
        print(student.to_string())


if __name__ == "__main__":
    main()
