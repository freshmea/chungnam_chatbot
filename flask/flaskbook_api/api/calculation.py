from os import abort
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import current_app, jsonify
from flaskbook_api.api.postprocess import draw_lines, draw_texts, make_color, make_line
from flaskbook_api.api.preparation import load_image
from flaskbook_api.api.preprocess import image_to_tensor

basedir = Path(__file__).parent.parent


def detection(request):
    dict_results = {}
    # 라벨 읽어 들이기
    labels = current_app.config["LABELS"]
    # 이미지 읽어 들이기
    image, filename = load_image(request)
    # 이미지 데이터를 텐서형의 수치 데이터로 변경
    image_tensor = image_to_tensor(image)

    # 학습 완료 모델의 읽어 들이기
    try:
        model = torch.load("model.pt")
    except FileNotFoundError:
        return jsonify("The model is not found"), 404

    # 모델의 추론 모드로 전환
    model = model.eval()
    # 추론의 실행
    output = model([image_tensor])[0]

    result_image = np.array(image.copy())
    # 학습 완료 모델이 검지한 물체의 이미지에 테투리 선과 라벨을 덧붙여 씀
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        # 점수가 0.6이상과 중복하지 않는 라벨로 좁힘
        if score > 0.6 and labels[label] not in dict_results:
            # 테두리 선의 색 결정
            color = make_color(labels)
            # 테두리 선의 작성
            line = make_line(result_image)
            # 검지 이미지의 테두리 선과 텍스트 라벨의 테두리 선의 위치 정보
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            # 이미지에 테두리 선을 덧붙여 씀
            draw_lines(c1, c2, result_image, line, color)
            # 이미지에 텍스트 라벨을 덧붙여 씀
            draw_texts(result_image, line, c1, color, labels[label])
            # 검지된 라벨과 점수의 사전을 작성
            dict_results[labels[label]] = round(100 * score.item())
    # 이미지 저장처의 디렉터리의 풀패스를 작성
    dir_image = str(basedir / "data" / "output" / filename)

    # 검지 후의 이미지 파일을 보존
    cv2.imwrite(dir_image, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return jsonify(dict_results), 201
