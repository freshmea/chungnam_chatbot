from pathlib import Path

import numpy as np
from flask import current_app
from PIL import Image


def get_grayscale(filenames: list[str]):
    """읽어 들인 손글씨 문자 이미지의 색을 그레이스케일로 변환하는 함수(그레이스케일은 색의 농담의 명함을 나누는 기법)"""
    dir_name = current_app.config["DIR_NAME"]
    dir_path = Path(__file__).resolve().parent.parent / dir_name
    for filename in filenames:
        img = Image.open(dir_path / filename).convert("L")
        yield img


def shrink_image(
    img, offset=5, crop_size: int = 8, pixel_size: int = 255, max_size: int = 16
):
    """이미지 크기를 8×8 픽셀의 크기로 통일하고, 명암도 16계조의 그레이스케일로 흑백으로 변환하는 함수"""
    img_array = np.asarray(img)
    h_indxis = np.where(img_array.min(axis=0) < 255)
    v_indxis = np.where(img_array.min(axis=1) < 255)
    h_min, h_max = h_indxis[0].min(), h_indxis[0].max()
    v_min, v_max = v_indxis[0].min(), v_indxis[0].max()
    width, hight = h_max - h_min, v_max - v_min

    if width > hight:
        center = (v_max + v_min) // 2
        left = h_min - offset
        upper = (center - width // 2) - 1 - offset
        right = h_max + offset
        lower = (center + width // 2) + offset
    else:
        center = (h_max + h_min + 1) // 2
        left = (center - hight // 2) - 1 - offset
        upper = v_min - offset
        right = (center + hight // 2) + offset
        lower = v_max + offset

    img_croped = img.crop((left, upper, right, lower)).resize((crop_size, crop_size))
    img_data256 = pixel_size - np.asarray(img_croped)

    min_bright, max_bright = img_data256.min(), img_data256.max()
    img_data16 = (img_data256 - min_bright) / (max_bright - min_bright) * max_size
    return img_data16


def get_shrinked_img(filenames: list[str]):
    """모델에 인풋하는 이미지의 수치 데이터의 리스트를 작성하는 함수"""
    img_test = np.empty((0, 64))
    for img in get_grayscale(filenames):
        img_data16 = shrink_image(img)
        img_test = np.r_[img_test, img_data16.astype(np.uint8).reshape(1, -1)]
    return img_test
