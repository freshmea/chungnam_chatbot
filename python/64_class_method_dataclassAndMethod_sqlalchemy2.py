# author: choi sugil
# date: 2023.10.25 version: 1.0.0 license: MIT brief: keyward
# description: 데이터클래스 와 클래스를 연동하는 프로그램 sqlite로 데이터를 읽어옴
from dataclasses import dataclass, field
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class Students(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    korean = Column(Integer)
    math = Column(Integer)
    english = Column(Integer)
    science = Column(Integer)


@dataclass
class StudentArg:
    name: str
    korean: int = 0
    math: int = 0
    english: int = 0
    science: int = 0


class Student:
    def __init__(self, arg: StudentArg | Students):
        """student class
        Args:
            args (StudentArg): Student dataclass

        Attributes:
            name: str
            korean: int
            math: int
            english: int
            science: int
            score_set: list
        """
        self.name = arg.name
        self.korean = arg.korean
        self.math = arg.math
        self.english = arg.english
        self.science = arg.science

    def get_sum(self):
        return sum((self.korean, self.math, self.english, self.science))

    def get_average(self):
        return self.get_sum() / 4

    def __str__(self):
        return f"{self.name}\t{self.get_sum()}\t{self.get_average():.2f}"


def main():
    students_list = []
    # make session sqlalchemy
    engine = create_engine("sqlite:///C:\chungnam_chatbot\python\sql_example.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    data = session.query(Students).all()
    for student in data:
        students_list.append(Student(student))

    print("이름\t총점\t평균")
    for student in students_list:
        print(student)
    session.commit()
    session.close()


if __name__ == "__main__":
    main()
