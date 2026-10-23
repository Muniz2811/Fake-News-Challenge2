FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/contraponto

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 bot

COPY --chown=bot:bot src/contraponto/ ./src/contraponto/

USER bot

# Perfil web opcional; o estágio final continua sendo o bot do Telegram.
FROM runtime AS web
USER root
COPY requirements/web.txt ./requirements/web.txt
RUN python -m pip install --no-cache-dir -r requirements/web.txt
COPY --chown=bot:bot assets/Logomarca.png assets/Logotipo.png assets/Wallpaper.png ./assets/
COPY --chown=bot:bot assets/CrimsonText-Regular.ttf assets/CrimsonText-OFL.txt ./assets/
USER bot
EXPOSE 8080
CMD ["python", "src/contraponto/web.py"]

FROM runtime AS bot
CMD ["python", "src/contraponto/bot.py"]
