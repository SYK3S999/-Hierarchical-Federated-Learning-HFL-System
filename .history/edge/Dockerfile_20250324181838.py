FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY edge.py .
COPY ../common common/
CMD ["python", "edge.py"]