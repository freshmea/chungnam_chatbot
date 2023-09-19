from flask_wtf.file import FileAllowed, FileField, FileRequired
from flask_wtf.form import FlaskForm
from wtforms.fields.simple import SubmitField


class UploadImageForm(FlaskForm):
    image = FileField(
        validators=[
            FileRequired("이미지 파일을 지정해 주세요."),
            FileAllowed(["jpg", "png", "jpeg"], "지원되지 않는 이미지 형식입니다."),
        ]
    )
    submit = SubmitField("업로드")
