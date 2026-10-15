"""Smoke HTTP dentro da imagem web, com APIs desativadas e sem rede externa."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    environment = dict(os.environ, PORT="8091", WEB_ACCESS_CODE="ci-room-code", LANGUAGE_RISK_BACKEND="tfidf")
    for key in ("OPENAI_API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "GOOGLE_FACT_CHECK_KEY", "TAVILY_API_KEY"):
        environment[key] = ""
    assert not list(Path("/app").rglob(".env*"))
    process = subprocess.Popen(
        [sys.executable, "src/contraponto/web.py"], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    clients = [urllib.request.build_opener(urllib.request.HTTPCookieProcessor()) for _ in range(4)]

    def request(client, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request("http://127.0.0.1:8091" + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with client.open(req, timeout=3) as response:
            return json.load(response)

    def wait_for(check, seconds=25):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("Servidor web encerrou antes da hora.")
            try:
                value = check()
                if value:
                    return value
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(.1)
        raise AssertionError("Timeout na verificação HTTP da imagem web.")

    try:
        wait_for(lambda: request(clients[0], "/healthz")["status"] == "ok")
        for index, client in enumerate(clients, 1):
            assert request(client, "/api/session", {
                "profile_id": f"p{index}", "access_code": "ci-room-code", "group_consent": True,
            })["id"] == f"p{index}"
        request(clients[0], "/api/messages", {"text": "A universidade publicou um estudo em 2026.", "chat_id": "group"})
        wait_for(lambda: request(clients[0], "/api/messages")["messages"][0]["status"] == "sent")
        for client in clients:
            message = request(client, "/api/messages")["messages"][0]
            assert message["analysis"] is None, "O grupo iniciou a verificação factual sem solicitação."
        mid = request(clients[0], "/api/messages")["messages"][0]["id"]
        request(clients[0], f"/api/messages/{mid}/verify", {})
        wait_for(lambda: request(clients[0], "/api/messages")["messages"][0]["status"] == "done")
        analysis = request(clients[0], "/api/messages")["messages"][0]["analysis"]
        assert analysis["claims"], "O fallback não extraiu a alegação de teste."
        assert all(claim["category"] in {"nao_verificada", "evidencia_insuficiente"} for claim in analysis["claims"])
        request(clients[0], "/api/messages", {"text": "Conversa privada de teste", "chat_id": "p2"})
        assert len(request(clients[1], "/api/messages?chat_id=p1")["messages"]) == 1
        assert request(clients[2], "/api/messages?chat_id=p1")["messages"] == []
        assert request(clients[2], "/api/messages?chat_id=bot")["messages"] == []
        # Encaminhamento explícito; só o bot de quem pediu recebe a investigação.
        private = request(clients[1], "/api/messages?chat_id=p1")["messages"][0]
        request(clients[1], f'/api/messages/{private["id"]}/forward', {"chat_id": "p1"})
        wait_for(lambda: request(clients[1], "/api/messages?chat_id=bot")["messages"][0]["status"] == "done")
        analysis = request(clients[1], "/api/messages?chat_id=bot")["messages"][0]["analysis"]
        assert analysis["claims"], "O fallback não extraiu a alegação de teste."
        assert all(claim["category"] in {"nao_verificada", "evidencia_insuficiente"} for claim in analysis["claims"])
        assert request(clients[0], "/api/messages?chat_id=bot")["messages"] == []
        for path in ("/", "/assets/app.js", "/assets/styles.css", "/brand/Logomarca.png", "/brand/Logotipo.png", "/brand/Wallpaper.png", "/brand/CrimsonText-Regular.ttf", "/brand/CrimsonText-OFL.txt"):
            with clients[0].open("http://127.0.0.1:8091" + path) as response:
                assert response.status == 200
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            logs, _ = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise AssertionError("SIGTERM não encerrou o servidor web.")
    assert process.returncode in (0, -signal.SIGTERM)
    assert "Application shutdown complete" in logs
    assert "ci-room-code" not in logs
    print("Web: quatro perfis, grupo, conversas privadas, fallback e SIGTERM: OK.")


if __name__ == "__main__":
    main()
