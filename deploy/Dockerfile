FROM python:3.11-slim

WORKDIR /app

COPY deploy/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and deployment code
COPY src/ src/
COPY deploy/ deploy/

# Copy committed checkpoint for demo
COPY src/configs/archive_results_configs/config_1/results/original/best_model.pth \
     src/configs/archive_results_configs/config_1/results/original/best_model.pth

ENV PYTHONPATH=/app
ENV KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 8501

CMD ["streamlit", "run", "deploy/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
