FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/contraponto

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 bot

COPY --chown=bot:bot src/contraponto/ ./src/contraponto/

USER bot
CMD ["python", "src/contraponto/bot.py"]
