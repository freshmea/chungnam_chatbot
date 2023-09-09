import pytest
from api.run import create_app, db


@pytest.fixture
def app():
    """
    테스트용의 데이터베이스를 작성
    """
    app = create_app()
    app_context = app.app_context()
    app_context.push()
    db.create_all()

    yield app

    db.session.remove()
    db.drop_all()
    app_context.pop()


@pytest.fixture
def client(app):
    """
    테스트용의 request 오브젝트를 작성
    """
    return app.test_client()
