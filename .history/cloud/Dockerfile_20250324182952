FROM python:3.9-slim
WORKDIR /app
COPY cloud/requirements.txt .
RUN pip install -r requirements.txt
COPY cloud/cloud.py .
COPY common/ common/
COPY data/ /data
CMD ["python", "cloud.py"]