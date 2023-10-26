# author: choi sugil
# date: 2023.10.25 version: 1.0.0 license: MIT brief: keyward
# description: 데이터클래스 와 클래스를 연동하는 프로그램 json 으로 데이터를 읽어옴
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StudentArg:
    name: str = ""
    korean: int = 0
    math: int = 0
    english: int = 0
    science: int = 0
    art: int = 0
    scores: dict = field(default_factory=dict)
    score_set: list = field(default_factory=list)

    @classmethod
    def from_json(cls, file):
        param_dir = r"C:\chungnam_chatbot\python"
        with open(Path(param_dir) / file, "r", encoding="UTF8") as f:
            data = json.loads(f.read())
        subjects = set()
        for item in data:
            subjects.update(item.keys())
        for subject in subjects:
            setattr(cls, subject, 0)
        print(subjects)
        print(data)
        return [cls(**item) for item in data]


students = StudentArg.from_json("params.json")
for student in students:
    print(student)
