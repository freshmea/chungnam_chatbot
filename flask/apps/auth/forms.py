from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, Length


class SignUpForm(FlaskForm):
    username = StringField(
        "사용자명",
        validators=[
            DataRequired("사용자명은 필수입니다."),
            Length(1, 30, "30문자 이내로 입력해 주세요."),
        ],
    )
    email = StringField(
        "메일 주소",
        validators=[
            DataRequired("메일 주소는 필수입니다."),
            Email("메일 주소의 형식으로 입력해 주세요."),
        ],
    )
    password = PasswordField("비밀번호", validators=[DataRequired("비밀번호는 필수입니다.")])
    submit = SubmitField("신규 등록")


class LoginForm(FlaskForm):
    email = StringField(
        "메일 주소",
        validators=[
            DataRequired("메일 주소는 필수입니다."),
            Email("메일 주소의 형식으로 입력하세요."),
        ],
    )
    password = PasswordField("비밀번호", validators=[DataRequired("비밀번호는 필수입니다.")])
    submit = SubmitField("로그인")
