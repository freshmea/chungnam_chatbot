def test_func1():
    assert 1 == 1


def test_func2():
    assert 2 == 2


# 픽스처의 함수를 인수로 지정하면 함수의 실행 결과가 건네진다
def test_func3(app_data):
    assert app_data == 3
