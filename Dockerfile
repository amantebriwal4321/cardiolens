FROM node:22-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-runtime.txt .
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --no-cache-dir -r requirements-runtime.txt

COPY . .
WORKDIR /app/web
RUN npm ci && npm run build

WORKDIR /app
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:${PATH}"
ENV BACKEND_URL=http://127.0.0.1:8091
EXPOSE 3921
CMD ["bash", "scripts/serve.sh"]
