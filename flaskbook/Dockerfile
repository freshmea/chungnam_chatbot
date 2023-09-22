FROM python:3.11.2

RUN apt-get update && apt-get install -y sqlite3 && apt-get install -y libsqlite3-dev

WORKDIR /usr/src

COPY ./apps /user/src/apps
COPY ./local.sqlite /user/src/local.sqlite
COPY ./requirements.txt /user/src/requirements.txt
COPY ./model.pt /usr/src/model.pt
COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install torch torchvision opencv-python
RUN pip install alembic==1.7.5
RUN pip install attrs==21.2.0
RUN pip install black==21.12b0
RUN pip install blinker==1.4
RUN pip install click==8.0.3
RUN pip install coverage==6.2
RUN pip install dnspython==2.1.0
RUN pip install email-validator==1.1.3
RUN pip install flake8==4.0.1
RUN pip install Flask==2.0.2
RUN pip install Flask-DebugToolbar==0.11.0
RUN pip install Flask-Login==0.5.0
RUN pip install Flask-Mail==0.9.1
RUN pip install Flask-Migrate==3.1.0
RUN pip install Flask-SQLAlchemy==2.5.1
RUN pip install Flask-WTF==1.0.0
RUN pip install idna==3.3
RUN pip install iniconfig==1.1.1
RUN pip install isort==5.10.1
RUN pip install itsdangerous==2.0.1
RUN pip install Jinja2==3.0.3
RUN pip install Mako==1.1.6
RUN pip install MarkupSafe==2.0.1
RUN pip install mccabe==0.6.1
RUN pip install mypy==0.920
RUN pip install mypy-extensions==0.4.3
RUN pip install numpy==1.21.4
RUN pip install opencv-python==4.5.4.60
RUN pip install packaging==21.3
RUN pip install pathspec==0.9.0
RUN pip install Pillow==8.4.0
RUN pip install platformdirs==2.4.0
RUN pip install pluggy==1.0.0
RUN pip install py==1.11.0
RUN pip install pycodestyle==2.8.0
RUN pip install pyflakes==2.4.0
RUN pip install pyparsing==3.0.6
RUN pip install pytest==6.2.5
RUN pip install pytest-cov==3.0.0
RUN pip install python-dotenv==0.19.2
RUN pip install SQLAlchemy==1.4.28
RUN pip install toml==0.10.2
RUN pip install tomli==1.2.3
RUN pip install typing_extensions==4.0.1
RUN pip install Werkzeug==2.0.2
RUN pip install WTForms==3.0.0


RUN echo "building..."

ENV FLASK_APP "apps.app:create_app('local')"
ENV IMAGE_URL "/storage/images/"

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0"]