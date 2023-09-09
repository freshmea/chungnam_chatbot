# ml_api

## Part 4로부터 시작하는 경우

### 프로젝트 설정

#### Mac/Linux

```
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -r requirements.txt
```

#### Windows（PowerShell）

```
> py -3 -m venv venv
> venv\Scripts\Activate.ps1
> pip install -r requirements.txt
```

### DB Migrate

```
(venv) $ flask db migrate
(venv) $ flask db upgrade
```

### 실행

```
(venv) flask run
```
