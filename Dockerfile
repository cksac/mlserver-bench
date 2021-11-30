from python:3.8

COPY ./requirement.txt /app/requirement.txt
RUN pip install -r /app/requirement.txt
