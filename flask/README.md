# Python Flask에 의한 Web 앱 개발 입문
## Git Clone

```
$ git clone https://github.com/ml-flaskbook/flaskbook.git
```

## 가상환경 만들기

### Mac/Linux

```
$ python3 -m venv venv
$ source venv/bin/activate
```

### Widows（PowerShell）

스크립트를 실행하기 위해 Windows PowerShell에 다음 명령으로 정책을 변경합니다.

```
> PowerShell Set-ExecutionPolicy RemoteSigned CurrentUser
```

정책을 변경한 다음 다음 명령을 실행합니다.

```
> py -m venv venv
> venv\Scripts\Activate.ps1
```

## 환경 파일 설치

```
$ cp -p .env.local .env
```

## 패키지 설치

```
(venv) $ pip install -r requirements.txt
```

## DB 마이그레이트

```
(venv) $ flask db init
(venv) $ flask db migrate
(venv) $ flask db upgrade
```

## 학습 모델 취득

```
(venv) $ python
>>> import torch
>>> import torchvision
>>> model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
>>> torch.save(model, "model.pt")
```

`model.pt`를 `apps/detector`로 이동

## 앱 실행

```
(venv) $ flask run
```

## 테스트 실행

```
$ pytest tests/detector
```

## 제2부로부터 시작하는 경우

다음 명령으로 제1부까지 기능을 설정

```
$ git checkout -b part1 tags/part1
```