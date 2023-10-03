from typing import Protocol

"""프로토콜은 클래스가 특정한 속성을 가지고 있음을 보장하는 데 사용됩니다.
pylance 를 사용하면 프로토콜을 사용하여 클래스가 특정한 속성을 가지고 있음을 보장할 수 있습니다.
"""


class Tproto(Protocol):
    def bark(self) -> str:
        ...

    def speak(self) -> str:
        ...


class T_dog:
    def bark(self) -> str:
        return "dog bark"

    def speak(self) -> str:
        return "Woof!"


class T_cat:
    def bark(self) -> str:
        return "cat bark"

    def speak(self) -> str:
        return "meaw!"


def make_animal_speak(animal: Tproto) -> str:
    return animal.speak()


def make_animal_bark(animal: Tproto) -> str:
    return animal.bark()


def main() -> None:
    dog = T_dog()
    cat = T_cat()

    make_animal_speak(animal=dog)
    make_animal_speak(animal=cat)


if __name__ == "__main__":
    main()
