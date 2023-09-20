FROM python:3.10.7

RUN apt-get update && apt-get install -y sqlite3 && apt-get install -y libsqlite3-dev

WORKDIR /usr/src

COPY ./apps /user/src/apps
COPY ./local.sqlite /user/src/local.sqlite
COPY ./requirements.txt /user/src/requirements.txt
COPY ./model.pt /usr/src/model.pt

RUN pip install --upgrade pip
RUN pip install torch torchvision opencv-python
RUN pip install -r requirments.txt

RUN echo "building..."

ENV FLASK_APP "apps.app:create_app('local')"
ENV IMAGE_URL "/storage/images/"

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0"]