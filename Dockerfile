from python:3.8

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
