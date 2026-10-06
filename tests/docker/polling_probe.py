"""Executado na imagem por bind mount; simula Telegram, nunca usa credenciais reais."""
import asyncio
import json
import os
from unittest.mock import patch

# O teste não lê .env e não aceita credenciais do computador/CI.
os.environ.update({
    "TELEGRAM_BOT_TOKEN": "123456:ci-fake-token",
    "OPENAI_API_KEY": "",
    "GOOGLE_FACT_CHECK_KEY": "",
    "TAVILY_API_KEY": "",
    "AUTHORIZED_GROUPS": "",
    "LANGUAGE_RISK_MODEL_PATH": "/app/models/nonexistent.joblib",
})

from telegram.request import HTTPXRequest

phase = 0
sent = []


def message(text, message_id):
    result = {
        "message_id": message_id, "date": 0,
        "chat": {"id": 42, "type": "private", "first_name": "Teste"},
        "from": {"id": 42, "is_bot": False, "first_name": "Teste"},
        "text": text,
    }
    if text == "/start":
        result["entities"] = [{"type": "bot_command", "offset": 0, "length": 6}]
    return result


async def fake_request(self, url, method, request_data=None, **kwargs):
    global phase
    endpoint = url.rsplit("/", 1)[-1]
    if endpoint == "getMe":
        result = {"id": 123456, "is_bot": True, "first_name": "Teste", "username": "ci_bot"}
    elif endpoint == "deleteWebhook":
        result = True
    elif endpoint == "getUpdates":
        if phase < 2:
            phase += 1
            text = "/start" if phase == 1 else "A universidade publicou um estudo sobre a qualidade da água."
            result = [{"update_id": phase, "message": message(text, phase)}]
        else:
            await asyncio.sleep(0.2)
            result = []
    elif endpoint == "sendMessage":
        text = request_data.parameters["text"]
        sent.append(text)
        result = message(text, len(sent) + 10)
        if "Relatório de Investigação" in text:
            assert any("Olá!" in item for item in sent), "/start não respondeu"
            assert any("investigação começou" in item for item in sent), "Sem confirmação inicial"
            assert "A universidade publicou um estudo" in text, "Fallback não extraiu a alegação"
            assert "insuficiente" in text, "Sem evidências deve haver abstenção"
            print("SMOKE_READY: start e fallback confirmados", flush=True)
    else:
        raise AssertionError(f"Chamada Telegram inesperada: {endpoint}")
    return 200, json.dumps({"ok": True, "result": result}).encode()


def main():
    with patch("dotenv.load_dotenv"), patch.object(HTTPXRequest, "do_request", fake_request):
        import bot
        assert bot.MotorIA([]).extrator_claims.client is None
        bot.main()
    print("SMOKE_STOPPED: polling encerrou normalmente", flush=True)


if __name__ == "__main__":
    main()
