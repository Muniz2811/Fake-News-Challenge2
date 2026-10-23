"""Sala web com sessões e motor simulado, sem APIs, chaves ou leitura de .env."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_search  # Isola configuração antes de importar o núcleo.
from fastapi.testclient import TestClient
from motor_ia import ResultadoAnalise
from verifier import VereditoAlegacao
from web import COOKIE, WebSettings, create_app, serialize_result


TEXT = "Texto confidencial usado apenas neste teste."


def result():
    return ResultadoAnalise(
        analise_linguistica="Score experimental 0,8; não é verificação factual.",
        score_linguistico=0.8,
        nome_modelo_linguistico="TF-IDF + regressão logística",
        vereditos=[
            VereditoAlegacao(1, "Primeira alegação", "apoiada", "Trecho da fonte A", ["https://example.org/a"]),
            VereditoAlegacao(2, "Segunda alegação", "conflitante", "Fontes divergentes", ["https://example.org/b"]),
        ],
        etapas_indisponiveis=["Uma fonte ficou indisponível."],
        requer_revisao=True,
    )


class TestWebRoom(unittest.TestCase):
    def setUp(self):
        self.motor = SimpleNamespace(
            analisar_texto=AsyncMock(return_value=result()),
            analisar_linguagem=AsyncMock(return_value=result()),
        )
        self.settings = WebSettings(access_code="código-de-teste", cooldown_seconds=0, queue_size=1)
        self.app = create_app(motor=self.motor, settings=self.settings)
        self.client = self.enterContext(TestClient(self.app))

    def join(self, profile_id="p1"):
        self.client.cookies.clear()  # Simula outro navegador sem apagar a sessão anterior.
        response = self.client.post("/api/session", json={"profile_id": profile_id, "access_code": "código-de-teste", "group_consent": True})
        self.assertEqual(response.status_code, 201)
        return {"Cookie": f"{COOKIE}={self.client.cookies.get(COOKIE)}"}, response.json()

    def finish(self):
        async def wait():
            await asyncio.wait_for(self.app.state.room.queue.join(), 3)
        self.client.portal.call(wait)

    def test_public_page_assets_and_health_expose_no_configuration(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Contra", response.text)
        self.assertIn('href="/brand/CrimsonText-Regular.ttf"', response.text)
        self.assertNotIn("typekit.net", response.text)
        self.assertIn("font-src 'self'", response.headers["content-security-policy"])
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertEqual(self.client.get("/assets/app.js").status_code, 200)
        self.assertEqual(self.client.get("/assets/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/.env").status_code, 404)
        config = self.client.get("/api/config").json()
        self.assertEqual(set(config), {"room_name", "code_required", "profiles"})
        self.assertTrue(config["code_required"])
        self.assertEqual([p["name"] for p in config["profiles"]], [f"Pessoa {n}" for n in range(1, 5)])
        self.motor.analisar_texto.assert_not_awaited()

    def test_brand_files_are_served_unchanged_without_exposing_other_files(self):
        root = Path(__file__).resolve().parents[1]
        for filename, mime in (
            ("Logomarca.png", "image/png"), ("Logotipo.png", "image/png"), ("Wallpaper.png", "image/png"),
            ("CrimsonText-Regular.ttf", "font/ttf"), ("CrimsonText-OFL.txt", "text/plain"),
        ):
            response = self.client.get(f"/brand/{filename}")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith(mime))
            self.assertEqual(response.content, (root / "assets" / filename).read_bytes())
        self.assertEqual(self.client.get("/brand/.env").status_code, 404)
        self.assertEqual(self.client.get("/brand/README.md").status_code, 404)

    def test_login_and_messages_require_access(self):
        self.assertEqual(self.client.get("/api/messages").status_code, 401)
        self.assertEqual(self.client.post("/api/messages", json={"text": TEXT}).status_code, 401)
        self.assertEqual(self.client.post("/api/messages/unknown/verify").status_code, 401)
        self.assertEqual(self.client.post("/api/messages/unknown/forward", json={"chat_id": "p1"}).status_code, 401)
        self.assertEqual(self.client.post("/api/session", json={"profile_id": "p1", "access_code": "errado", "group_consent": True}).status_code, 403)
        self.motor.analisar_texto.assert_not_awaited()

    def test_session_cookie_logout_and_expiration(self):
        headers, user = self.join()
        self.assertEqual(self.client.get("/api/session", headers=headers).json(), user)
        response = self.client.post("/api/session", json={"profile_id": "p1", "access_code": "código-de-teste", "group_consent": True})
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("SameSite=strict", response.headers["set-cookie"])
        self.assertEqual(self.client.get("/api/session", headers=headers).status_code, 401)
        token = self.client.cookies.get(COOKIE)
        self.app.state.room.sessions[token].expires = 0
        self.assertEqual(self.client.get("/api/messages").status_code, 401)
        self.join()
        self.assertEqual(self.client.delete("/api/session").status_code, 204)
        self.assertEqual(self.client.get("/api/session").status_code, 401)

    def test_two_people_share_messages_and_per_claim_sources(self):
        pedro, user_pedro = self.join("p1")
        maria, user_maria = self.join("p2")
        self.assertNotEqual(user_pedro["id"], user_maria["id"])
        response = self.client.post("/api/messages", headers=pedro, json={"text": TEXT})
        self.assertEqual(response.status_code, 202)
        self.finish()
        screened = self.client.get("/api/messages", headers=pedro).json()["messages"][0]
        self.assertEqual(screened["status"], "sent")
        self.assertIsNone(screened["analysis"])
        self.motor.analisar_texto.assert_not_awaited()
        self.motor.analisar_linguagem.assert_not_awaited()
        self.assertEqual(self.client.post(f'/api/messages/{screened["id"]}/verify', headers=maria).status_code, 202)
        self.finish()
        first = self.client.get("/api/messages", headers=pedro).json()
        second = self.client.get("/api/messages", headers=maria).json()
        self.assertEqual(first["messages"], second["messages"])
        message = second["messages"][0]
        self.assertEqual(message["author"], "Pessoa 1")
        self.assertEqual(message["status"], "done")
        self.assertEqual(message["analysis"]["claims"][0]["sources"], ["https://example.org/a"])
        self.assertEqual(message["analysis"]["claims"][1]["sources"], ["https://example.org/b"])
        self.assertEqual(message["analysis"]["claims"][1]["category"], "conflitante")
        self.assertEqual(message["analysis"]["language"]["score"], .8)
        self.motor.analisar_texto.assert_awaited_once_with(TEXT)
        self.assertEqual({member["name"] for member in second["profiles"] if member["online"]}, {"Pessoa 1", "Pessoa 2"})
        unchanged = self.client.get(f'/api/messages?version={second["version"]}', headers=maria).json()
        self.assertIsNone(unchanged["messages"])

    def test_invalid_messages_and_spoofed_fields_are_rejected(self):
        self.join()
        for body in ({"text": " "}, {"text": ""}, {"text": 123}, {"text": "a" * 4001},
                     {"text": TEXT, "author": "Outra pessoa"}, {"text": TEXT, "modelo_id": "bertimbau_mlp"}):
            with self.subTest(body=list(body)):
                self.assertEqual(self.client.post("/api/messages", json=body).status_code, 422)
        self.motor.analisar_texto.assert_not_awaited()

    def test_queue_is_bounded_and_motor_calls_are_serial(self):
        gate = asyncio.Event()

        async def slow(text):
            await gate.wait()
            return result()

        self.motor.analisar_texto.side_effect = slow
        first, _ = self.join("p1")
        second, _ = self.join("p2")
        third, _ = self.join("p3")

        def request_analysis(headers, text):
            mid = self.client.post("/api/messages", headers=headers, json={"text": text}).json()["id"]
            return self.client.post(f"/api/messages/{mid}/verify", headers=headers)

        try:
            self.assertEqual(request_analysis(first, TEXT).status_code, 202)
            self.assertEqual(request_analysis(first, TEXT).status_code, 429)
            self.assertEqual(request_analysis(second, "Outra mensagem").status_code, 202)
            self.assertEqual(request_analysis(third, "Fila cheia").status_code, 429)
            # Conversar continua possível mesmo durante uma investigação.
            for chat in ("group", "p2"):
                self.assertEqual(self.client.post("/api/messages", headers=first,
                                                 json={"text": "Conversa durante a análise", "chat_id": chat}).status_code, 202)
            self.assertEqual(self.motor.analisar_texto.await_count, 1)
        finally:
            self.client.portal.call(gate.set)
            self.finish()
        self.assertEqual(self.motor.analisar_texto.await_count, 2)
        self.motor.analisar_linguagem.assert_not_awaited()
        self.assertEqual(sum(message["status"] == "done" for message in self.app.state.room.messages["group"]), 2)

    def test_engine_failure_keeps_message_and_allows_next_analysis_without_payload_in_logs(self):
        self.join()
        self.motor.analisar_texto.side_effect = [RuntimeError(TEXT), result()]
        mid = self.client.post("/api/messages", json={"text": TEXT}).json()["id"]
        with self.assertLogs("web", level="ERROR") as captured:
            self.client.post(f"/api/messages/{mid}/verify")
            self.finish()
        self.assertNotIn(TEXT, " ".join(captured.output))
        self.assertEqual(self.app.state.room.messages["group"][0]["status"], "error")
        self.assertEqual(self.app.state.room.messages["group"][0]["text"], TEXT)
        mid = self.client.post("/api/messages", json={"text": "Uma nova mensagem"}).json()["id"]
        self.client.post(f"/api/messages/{mid}/verify")
        self.finish()
        self.assertEqual(self.app.state.room.messages["group"][1]["status"], "done")

    def test_history_retains_only_the_configured_limit(self):
        self.app.state.room.settings = replace(self.settings, max_messages=2)
        self.join()
        for index in range(3):
            self.client.post("/api/messages", json={"text": f"Mensagem {index}"})
            self.finish()
        self.assertEqual([message["text"] for message in self.app.state.room.messages["group"]], ["Mensagem 1", "Mensagem 2"])

    def test_cooldown_and_session_limit_are_enforced(self):
        self.app.state.room.settings = replace(self.settings, cooldown_seconds=60, max_sessions=1)
        self.join()
        first = self.client.post("/api/messages", json={"text": TEXT}).json()["id"]
        second = self.client.post("/api/messages", json={"text": TEXT}).json()["id"]
        self.assertEqual(self.client.post(f"/api/messages/{first}/verify").status_code, 202)
        self.finish()
        self.assertEqual(self.client.post(f"/api/messages/{second}/verify").status_code, 429)
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/api/session", json={"profile_id": "p2", "access_code": "código-de-teste", "group_consent": True}).status_code, 503)

    def test_web_initializes_motor_once_without_requiring_telegram_token(self):
        with patch("web.MotorIA", return_value=self.motor) as factory, patch("config.TELEGRAM_BOT_TOKEN", None):
            with TestClient(create_app(settings=WebSettings())) as client:
                self.assertEqual(client.get("/healthz").status_code, 200)
                self.assertEqual(client.post("/api/session", json={"profile_id": "p1", "group_consent": True}).status_code, 201)
            factory.assert_called_once_with()

    def test_result_preserves_abstention_warnings_and_unavailable_score(self):
        value = result()
        value.score_linguistico = float("nan")
        value.vereditos[0].categoria = "nao_verificada"
        value.vereditos[0].fontes_citadas = []
        output = serialize_result(value)
        self.assertIsNone(output["language"]["score"])
        self.assertEqual(output["claims"][0]["label"], "não verificada")
        self.assertEqual(output["claims"][0]["sources"], [])
        self.assertEqual(output["warnings"], value.etapas_indisponiveis)

    def test_only_four_profiles_and_each_is_exclusive_until_logout_or_inactivity(self):
        users = [self.join(f"p{index}")[0] for index in range(1, 5)]
        self.client.cookies.clear()
        for profile in ("p1", "p2", "p3", "p4"):
            response = self.client.post("/api/session", json={"profile_id": profile, "access_code": self.settings.access_code, "group_consent": True})
            self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.post("/api/session", json={"profile_id": "p5", "access_code": self.settings.access_code, "group_consent": True}).status_code, 422)
        self.client.delete("/api/session", headers=users[0])
        self.join("p1")
        token = self.client.cookies.get(COOKIE)
        self.app.state.room.sessions[token].seen -= 61
        self.join("p1")
        self.assertEqual(len(self.app.state.room.sessions), 4)

    def test_private_chat_is_visible_only_to_its_two_profiles_and_never_analyzed(self):
        one, _ = self.join("p1")
        two, _ = self.join("p2")
        three, _ = self.join("p3")
        response = self.client.post("/api/messages", headers=one, json={"text": TEXT, "chat_id": "p2"})
        self.assertEqual(response.status_code, 202)
        received = self.client.get("/api/messages?chat_id=p1", headers=two).json()["messages"]
        self.assertEqual(received[0]["text"], TEXT)
        self.assertEqual(received[0]["status"], "sent")
        self.assertEqual(self.client.get("/api/messages?chat_id=p1", headers=three).json()["messages"], [])
        self.assertEqual(self.client.get("/api/messages?chat_id=p2", headers=three).json()["messages"], [])
        self.assertEqual(self.client.get("/api/messages?chat_id=dm:p1:p2", headers=three).status_code, 404)
        self.assertEqual(self.client.get("/api/messages", headers=three).json()["messages"], [])
        self.motor.analisar_texto.assert_not_awaited()

    def test_bot_chat_is_private_and_uses_same_motor(self):
        one, _ = self.join("p1")
        two, _ = self.join("p2")
        self.client.post("/api/messages", headers=one, json={"text": TEXT, "chat_id": "bot"})
        self.finish()
        self.motor.analisar_texto.assert_awaited_once_with(TEXT)
        result = self.client.get("/api/messages?chat_id=bot", headers=one).json()["messages"]
        self.assertEqual(result[0]["analysis"]["claims"][0]["label"], "apoiada")
        self.assertEqual(self.client.get("/api/messages?chat_id=bot", headers=two).json()["messages"], [])
        self.assertEqual(self.client.get("/api/messages?chat_id=bot:p1", headers=two).status_code, 404)
        self.assertEqual(self.client.get("/api/messages", headers=two).json()["messages"], [])

    def test_chat_list_contains_group_bot_and_only_other_three_people(self):
        self.join("p2")
        chats = self.client.get("/api/chats").json()
        self.assertEqual([chat["id"] for chat in chats], ["group", "bot", "p1", "p3", "p4"])
        self.assertEqual(self.client.post("/api/messages", json={"text": TEXT, "chat_id": "p2"}).status_code, 404)

    def test_join_requires_explicit_group_consent(self):
        for consent in (None, False, "true"):
            body = {"profile_id": "p1", "access_code": self.settings.access_code}
            if consent is not None:
                body["group_consent"] = consent
            self.assertIn(self.client.post("/api/session", json=body).status_code, (403, 422))
        self.assertEqual(self.app.state.room.sessions, {})
        self.motor.analisar_linguagem.assert_not_awaited()
        self.motor.analisar_texto.assert_not_awaited()

    def test_all_group_messages_wait_for_click_without_calling_any_analyzer(self):
        self.join()
        for text in ("oi", "Tudo bem?", TEXT):
            self.assertEqual(self.client.post("/api/messages", json={"text": text}).status_code, 202)
        self.finish()
        messages = self.client.get("/api/messages").json()["messages"]
        self.assertEqual([item["text"] for item in messages], ["oi", "Tudo bem?", TEXT])
        self.assertTrue(all(item["status"] == "sent" and item["analysis"] is None for item in messages))
        self.motor.analisar_texto.assert_not_awaited()
        self.motor.analisar_linguagem.assert_not_awaited()
        self.assertEqual(self.client.post(f'/api/messages/{messages[0]["id"]}/verify').status_code, 202)
        self.finish()
        self.motor.analisar_texto.assert_awaited_once_with("oi")

    def test_explicit_group_verification_deduplicates_and_releases_requester_not_author(self):
        author, _ = self.join("p1")
        requester, _ = self.join("p2")
        mid = self.client.post("/api/messages", headers=author, json={"text": TEXT}).json()["id"]
        self.finish()
        gate = asyncio.Event()

        async def slow(text):
            await gate.wait()
            return result()

        self.motor.analisar_texto.side_effect = slow
        try:
            self.assertEqual(self.client.post(f"/api/messages/{mid}/verify", headers=requester).status_code, 202)
            self.assertEqual(self.client.post(f"/api/messages/{mid}/verify", headers=author).status_code, 409)
            self.assertIn("p2", self.app.state.room.pending_users)
            self.assertNotIn("p1", self.app.state.room.pending_users)
        finally:
            self.client.portal.call(gate.set)
            self.finish()
        self.assertFalse(self.app.state.room.pending_users)
        self.motor.analisar_texto.assert_awaited_once_with(TEXT)
        self.assertEqual(self.client.post(f"/api/messages/{mid}/verify", headers=requester).status_code, 409)

    def test_private_forward_is_explicit_authorized_and_result_stays_in_requesters_bot(self):
        one, _ = self.join("p1")
        two, _ = self.join("p2")
        three, _ = self.join("p3")
        self.client.post("/api/messages", headers=one, json={"text": "Outro texto privado", "chat_id": "p2"})
        mid = self.client.post("/api/messages", headers=one, json={"text": TEXT, "chat_id": "p2"}).json()["id"]
        self.motor.analisar_linguagem.assert_not_awaited()
        self.motor.analisar_texto.assert_not_awaited()
        self.assertEqual(self.client.post(f"/api/messages/{mid}/verify", headers=two).status_code, 404)
        self.assertEqual(self.client.post(f"/api/messages/{mid}/forward", headers=three, json={"chat_id": "p1"}).status_code, 404)
        self.assertEqual(self.client.post(f"/api/messages/{mid}/forward", headers=three, json={"chat_id": "dm:p1:p2"}).status_code, 422)
        response = self.client.post(f"/api/messages/{mid}/forward", headers=two, json={"chat_id": "p1"})
        self.assertEqual(response.status_code, 202)
        self.finish()
        self.motor.analisar_texto.assert_awaited_once_with(TEXT)
        self.motor.analisar_linguagem.assert_not_awaited()
        bot = self.client.get("/api/messages?chat_id=bot", headers=two).json()["messages"]
        self.assertEqual(len(bot), 1)
        self.assertEqual(bot[0]["text"], TEXT)
        self.assertIsNotNone(bot[0]["analysis"])
        self.assertEqual(bot[0]["author_id"], "p2")
        self.assertEqual(self.client.get("/api/messages?chat_id=bot", headers=one).json()["messages"], [])
        self.assertEqual(self.client.get("/api/messages?chat_id=bot", headers=three).json()["messages"], [])
        private = self.client.get("/api/messages?chat_id=p1", headers=two).json()["messages"]
        self.assertTrue(all(msg["status"] == "sent" and msg["analysis"] is None for msg in private))
        self.assertEqual(self.client.get("/api/messages", headers=one).json()["messages"], [])

    def test_group_verification_failure_can_be_retried(self):
        self.join()
        mid = self.client.post("/api/messages", json={"text": TEXT}).json()["id"]
        self.finish()
        self.motor.analisar_texto.side_effect = [RuntimeError(TEXT), result()]
        with self.assertLogs("web", level="ERROR") as captured:
            self.assertEqual(self.client.post(f"/api/messages/{mid}/verify").status_code, 202)
            self.finish()
        message = self.client.get("/api/messages").json()["messages"][0]
        self.assertEqual(message["status"], "error")
        self.assertEqual(message["text"], TEXT)
        self.assertNotIn(TEXT, " ".join(captured.output))
        self.assertEqual(self.client.post(f"/api/messages/{mid}/verify").status_code, 202)
        self.finish()
        self.assertEqual(self.client.get("/api/messages").json()["messages"][0]["status"], "done")
