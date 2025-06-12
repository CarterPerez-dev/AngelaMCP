FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    redis-tools \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $APP_HOME

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -ms /bin/bash angelamcp
RUN chown -R angelamcp:angelamcp $APP_HOME

USER angelamcp

CMD ["python", "-m", "src.main"]
