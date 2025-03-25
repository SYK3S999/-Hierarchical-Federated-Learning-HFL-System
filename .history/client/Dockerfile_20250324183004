FROM python:3.9-slim
WORKDIR /app
COPY client/requirements.txt .
RUN pip install -r requirements.txt
COPY client/client.py .
COPY common/ common/
COPY data/ /data
CMD ["python", "client.py"]