# author: choi sugil
# date: 2023.10.25 version: 1.0.0 license: MIT brief: keyward
# description: 데이터클래스 와 클래스를 연동하는 프로그램 sqlite로 데이터를 읽어옴
from dataclasses import dataclass, field
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

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
    score_set: list = field(default_factory=list)

    def __post_init__(self):
        self.score_set = [self.korean, self.math, self.english, self.science]


class Student:
    def __init__(self, arg: StudentArg):
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
        self.score_set = arg.score_set

    def get_sum(self):
        return sum(self.score_set)

    def get_average(self):
        return self.get_sum() / 4

    def __str__(self):
        return f"{self.name}\t{self.get_sum()}\t{self.get_average():.2f}"


def main():
    students = []
    # make session sqlalchemy
    engine = create_engine("sqlite:///C:\chungnam_chatbot\python\sql_example.db")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    session = Session()
    session.add_all(
        [
            Students(name="연하진", korean=92, math=98, english=96, science=98),
            Students(name="구지연", korean=58, math=96, english=78, science=90),
            Students(name="나선주", korean=96, math=92, english=100, science=92),
            Students(name="윤아린", korean=95, math=98, english=98, science=98),
            Students(name="윤명월", korean=64, math=88, english=92, science=92),
            Students(name="김미화", korean=82, math=86, english=98, science=88),
            Students(name="김연화", korean=88, math=74, english=78, science=92),
            Students(name="박아현", korean=97, math=92, english=88, science=95),
            Students(name="서준서", korean=45, math=52, english=72, science=78),
            Students(name="이준기", korean=66, math=58, english=68, science=72),
        ]
    )
    session.commit()
    session.close()
    # students = session.query(Students).all()
    # for student in students:
    # print(student)
    # arg = StudentArg(*row)
    # students.append(Student(arg))
    # print(f"{students[-1].name} : {students[-1].score_set}")

    print("이름\t총점\t평균")
    for student in students:
        print(student)


if __name__ == "__main__":
    main()
