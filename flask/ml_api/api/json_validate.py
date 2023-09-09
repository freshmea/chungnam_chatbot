from functools import wraps

from flask import current_app, jsonify, request
from jsonschema import ValidationError, validate
from werkzeug.exceptions import BadRequest


def validate_json(f):
    @wraps(f)
    def wrapper(*args, **kw):
        # 요청의 컨텐츠 타입이 JSON인지 여부를 체크한다
        ctype = request.headers.get("Content-Type")
        method_ = request.headers.get("X-HTTP-Method-Override", request.method)
        if method_.lower() == request.method.lower() and "json" in ctype:
            try:
                # body 메시지가 애당초 있는지 여부를 체크한다
                request.json
            except BadRequest as e:
                msg = "This is an invalid json"
                return jsonify({"error": msg}), 400
            return f(*args, **kw)

    return wrapper


def validate_schema(schema_name):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            try:
                # 조금 전, 정의한 json 파일대로 json의 body 메시지가 보내졌는지 여부를 체크한다
                validate(request.json, current_app.config[schema_name])
            except ValidationError as e:
                return jsonify({"error": e.message}), 400
            return f(*args, **kw)

        return wrapper

    return decorator
