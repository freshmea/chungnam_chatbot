import uuid
from pathlib import Path

from flask import abort, current_app, jsonify
from sqlalchemy.exc import SQLAlchemyError

from api.models import ImageInfo, db


def load_filenames(dir_name: str) -> list[str]:
    """손글씨 문자 이미지가 놓여있는 패스로부터 파일명을 취득하고, 리스트를 작성"""
    included_ext = current_app.config["INCLUDED_EXTENTION"]
    dir_path = Path(__file__).resolve().parent.parent / dir_name
    files = Path(dir_path).iterdir()
    filenames = sorted(
        [
            Path(str(file)).name
            for file in files
            if Path(str(file)).suffix in included_ext
        ]
    )
    return filenames


def insert_filenames(request) -> tuple:
    """손글씨 문자 이미지의 파일명을 데이터베이스에 보존"""
    dir_name = request.json["dir_name"]
    filenames = load_filenames(dir_name)
    file_id = str(uuid.uuid4())
    for filename in filenames:
        db.session.add(ImageInfo(file_id=file_id, filename=filename))
    try:
        db.session.commit()
    except SQLAlchemyError as error:
        db.session.rollback()
        abort(500, {"error_message": str(error)})
    return jsonify({"file_id": file_id}), 201


def extract_filenames(file_id: str) -> list[str]:
    """손글씨 문자 이미지의 파일명을 데이터베이스로부터 취득"""
    img_obj = db.session.query(ImageInfo).filter(ImageInfo.file_id == file_id)
    filenames = [img.filename for img in img_obj if img.filename]
    if not filenames:
        # p448: abort로 처리를 멈추는 경우의 코드
        # abort(404, {"error_message": "filenames are not found in database"})

        # p449: abort로 처리를 멈추지 않고, jsonify를 구현한 경우의 코드
        return (
            jsonify({"message": "filenames are not found in database", "result": 400}),
            400,
        )

    return filenames
