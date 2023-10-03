from python.Aprotocol import (
    Tproto,
    T_dog,
    T_cat,
    make_animal_speak,
)


def test_true() -> None:
    assert True


def test_Tproto_bark() -> None:
    """Tproto.bark() 메서드가 존재하는지 확인합니다."""
    assert hasattr(Tproto, "bark")
    assert callable(Tproto.bark)


def test_Tproto_speak() -> None:
    """Tproto.speak() 메서드가 존재하는지 확인합니다."""
    assert hasattr(Tproto, "speak")
    assert callable(Tproto.speak)


def test_T_dog_bark() -> None:
    """T_dog.bark() 메서드가 존재하는지 확인합니다."""
    assert hasattr(T_dog, "bark")
    assert callable(T_dog.bark)


def test_T_dog_speak() -> None:
    """T_dog.speak() 메서드가 존재하는지 확인합니다."""
    assert hasattr(T_dog, "speak")
    assert callable(T_dog.speak)


def test_T_cat_bark() -> None:
    """T_cat.bark() 메서드가 존재하는지 확인합니다."""
    assert hasattr(T_cat, "bark")
    assert callable(T_cat.bark)


def test_T_cat_speak() -> None:
    """T_cat.speak() 메서드가 존재하는지 확인합니다."""
    assert hasattr(T_cat, "speak")
    assert callable(T_cat.speak)


def test_make_animal_speak() -> None:
    """make_animal_speak() 함수가 존재하는지 확인합니다."""
    assert hasattr(make_animal_speak, "__annotations__")
    """ make_animal_speak() 함수가 T_dog() 인스턴스를 받아들일 수 있는지 확인합니다."""
    assert isinstance(make_animal_speak(T_dog()), str)
    """ make_animal_speak() 함수가 T_cat() 인스턴스를 받아들일 수 있는지 확인합니다."""
    assert isinstance(make_animal_speak(T_cat()), str)
    """ make_animal_speak() 함수가 T_dog() 로 출력이 가능한지 확인합니다."""
    assert make_animal_speak(T_dog()) == "Woof!"
    """ make_animal_speak() 함수가 T_cat() 로 출력이 가능한지 확인합니다."""
    assert make_animal_speak(T_cat()) == "meaw!"
