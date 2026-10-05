"""Regressão de envio do relatório com caracteres reservados, sem APIs reais."""
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

import test_search  # Impede leitura de .env e chamadas externas.
from bot import BotTelegram, _configurar_logging
from claims import Alegacao
from motor_ia import ResultadoAnalise
from search import Evidencia


def resultado():
    return ResultadoAnalise(
        analise_linguistica="Score experimental: 0,428.",
        alegacoes=[Alegacao(1, "O estudo_2026 diz *algo* [pendente]", "trecho_original")],
        resultado_factual="Conclusão: não verificada\nResposta inválida ou indisponível.",
        evidencias=[Evidencia(
            title="Fonte", url="https://example.org/estudo_2026?q=[agua]",
            publisher="Publicador", snippet="Texto",
        )],
    )


class TestBotReport(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = BotTelegram.__new__(BotTelegram)
        self.result = resultado()
        self.bot.motor = SimpleNamespace(analisar_texto=AsyncMock(return_value=self.result))
        self.message = SimpleNamespace(
            text="Texto de teste", chat=SimpleNamespace(type="private"), reply_text=AsyncMock(),
        )
        self.update = SimpleNamespace(message=self.message)

    async def test_private_report_is_sent_without_entity_parser(self):
        await self.bot._handle_private_message(self.update, SimpleNamespace())
        calls = self.message.reply_text.await_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("investigação começou", calls[0].args[0])
        self.assertEqual(calls[1].args[0], self.bot._formatar_relatorio(self.result))
        self.assertNotIn("parse_mode", calls[1].kwargs)
        self.assertNotIn("entities", calls[1].kwargs)
        self.assertIn(self.result.alegacoes[0].texto, calls[1].args[0])
        self.assertIn(self.result.evidencias[0].url, calls[1].args[0])
        self.assertNotIn("Alegacao(", calls[1].args[0])

    async def test_deep_link_uses_the_same_safe_format(self):
        self.bot._fetch = lambda _id: self.result
        await self.bot._cmd_start(self.update, SimpleNamespace(args=["test-id"]))
        self.message.reply_text.assert_awaited_once_with(self.bot._formatar_relatorio(self.result))


class TestLogging(unittest.TestCase):
    def test_http_request_info_logs_disabled(self):
        names = ("httpx", "httpx2", "httpcore")
        previous = {name: logging.getLogger(name).level for name in names}
        self.addCleanup(lambda: [logging.getLogger(name).setLevel(level) for name, level in previous.items()])
        _configurar_logging()
        for name in names:
            self.assertFalse(logging.getLogger(name).isEnabledFor(logging.INFO))
            self.assertTrue(logging.getLogger(name).isEnabledFor(logging.WARNING))
