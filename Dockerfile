# Base image with python
FROM python:3.11-slim

WORKDIR /app

# install system dependencies needed by tensorflow and pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first, leverages docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . .

# expose both api and streamlit ports
EXPOSE 8000 8501

# default command runs the api, overridden for ui container via docker-compose
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]