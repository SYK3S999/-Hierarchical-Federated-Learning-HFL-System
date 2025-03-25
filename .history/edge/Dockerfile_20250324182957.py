FROM python:3.9-slim
WORKDIR /app
COPY edge/requirements.txt .
RUN pip install -r requirements.txt
COPY edge/edge.py .
COPY common/ common/
CMD ["python", "edge.py"]