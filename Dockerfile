FROM python:3.8

ADD inference.py .

RUN pip install -r rqeuirements.txt

RUN mkdir /data

CMD ["python", "./inference.py runserver 5000"]
