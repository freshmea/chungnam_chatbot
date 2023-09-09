import os
import shutil

import pytest
from apps.app import create_app, db
from apps.crud.models import User
from apps.detector.models import UserImage, UserImageTag


# 픽스처 함수를 작성한다
@pytest.fixture
def fixture_app():
    # 셋업 처리
    # 테스트용의 컨피그를 사용하기 위해서 인수에 testing을 지정한다
    app = create_app("testing")

    # 데이터베이스를 이용하기 위한 선언을 한다
    app.app_context().push()

    # 테스트용 데이터베이스의 테이블을 작성한다
    with app.app_context():
        db.create_all()

    # 테스트용의 이미지 업로드 디렉터리를 작성한다
    os.mkdir(app.config["UPLOAD_FOLDER"])

    # 테스트를 실행한다
    yield app

    # 클린업 처리
    # user 테이블의 레코드를 삭제한다
    User.query.delete()

    # image 테이블의 레코드를 삭제한다
    UserImage.query.delete()

    # user_image_tags 테이블의 레코드를 삭제한다
    UserImageTag.query.delete()

    # 테스트용의 이미지 업로드 디렉터리를 삭제한다
    shutil.rmtree(app.config["UPLOAD_FOLDER"])

    db.session.commit()


# Flask의 테스트 클라이언트를 반환하는 픽스처 함수를 작성한다
@pytest.fixture
def client(fixture_app):
    # Flask의 테스트용 클라이언트를 반환한다
    return fixture_app.test_client()
