FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 MLFLOW_DISABLE_AGENT_HINT=1 MPLCONFIGDIR=/tmp/mpl SEMICONDUCTOR_STATE_DIR=/app/semiconductor_state
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY semiconductor/ semiconductor/
COPY data/semiconductor_etch_timeseries_3years.csv data/semiconductor_etch_timeseries_3years.csv
# Train/register in the container so local absolute artifact paths are never reused.
RUN python -m semiconductor.train --register && python -c "from semiconductor.runtime import ModelManager; ModelManager('mlflow').get()"
ENV MODEL_SOURCE=mlflow LOADING_MODE=eager
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["python", "-m", "uvicorn", "semiconductor.app:app", "--host", "0.0.0.0", "--port", "8000"]
