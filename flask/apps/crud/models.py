from datetime import datetime

from apps.app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash


# db.Model을 상속한 User 클래스를 작성한다
class User(db.Model, UserMixin):
    # 테이블명을 지정한다
    __tablename__ = "users"
    # 컬럼을 정의한다
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, index=True)
    email = db.Column(db.String, unique=True, index=True)
    password_hash = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # backref를 이용하여 relation 정보를 설정한다
    user_images = db.relationship("UserImage", backref="user")

    # 비밀번호를 세트하기 위한 프로퍼티
    @property
    def password(self):
        raise AttributeError("읽어 들일 수 없음")

    # 비밀번호를 세트하기 위한 센터 함수로 해시화한 비밀번호를 세트한다
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    # 비밀번호 체크를 한다
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    # 메일 주소 중복 체크를 한다
    def is_duplicate_email(self):
        return User.query.filter_by(email=self.email).first() is not None


# 로그인하고 있는 사용자 정보를 취득하는 함수를 작성한다
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)
