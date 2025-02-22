FROM python:3.9-slim

WORKDIR /Website-code

COPY Website-code/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Website-code/ .

EXPOSE 5000

ENV FLASK_APP=Website-code/app.py

CMD ["python", "app.py"]
