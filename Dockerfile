FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
EXPOSE 7860
EXPOSE 7861

CMD ["python", "app.py"]
