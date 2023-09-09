from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ImageInfo(db.Model):
    # 테이블 정의
    __tablename__ = "image_info"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_id = db.Column(db.String)
    filename = db.Column(db.String)

    def __repr__(self):
        return f"<Filename {self.filename}>"
