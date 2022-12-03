FROM python:3.9.14-alpine3.16

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000