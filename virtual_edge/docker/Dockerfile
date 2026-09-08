FROM python:3.11-slim-bookworm

LABEL project="FloodTwin-HIL"
LABEL target="Virtual Raspberry Pi 4B"
LABEL org.opencontainers.image.title="FloodTwin-HIL Virtual Edge"
LABEL org.opencontainers.image.description="Research Simulation Execution Layer For FloodTwin-HIL"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget nano vim htop procps \
    net-tools iputils-ping ca-certificates lm-sensors \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/FloodTwin-HIL

COPY virtual_edge/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/app/FloodTwin-HIL

CMD ["/bin/bash"]