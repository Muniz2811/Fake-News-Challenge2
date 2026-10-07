"""Regressão de envio do relatório com caracteres reservados, sem APIs reais."""
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

import test_search  # noqa: F401  # Impede leitura de .env e chamadas externas.
from bot import BotTelegram, _configurar_logging
from claims import Alegacao
from motor_ia import ResultadoAnalise
from search import Evidencia
from verifier import VereditoAlegacao


def resultado():
    return ResultadoAnalise(
        texto_original="O estudo de 2026 apresentou um resultado.",
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
        chamada = self.message.reply_text.await_args
        self.assertEqual(chamada.args[0], self.bot._formatar_relatorio(self.result))
        botao = chamada.kwargs["reply_markup"].inline_keyboard[0][0]
        self.assertEqual(botao.callback_data, "socratico:test-id")

    async def test_long_report_is_split_in_order_within_limit(self):
        self.result.vereditos = [
            veredito(i, "refutada", "Justificativa longa. " * 40, [f"https://exemplo.org/{i}"])
            for i in range(1, 41)
        ]
        self.bot._fetch = lambda _id: self.result
        await self.bot._cmd_start(self.update, SimpleNamespace(args=["test-id"]))
        partes = [c.args[0] for c in self.message.reply_text.await_args_list]
        self.assertGreater(len(partes), 1)
        chamadas = self.message.reply_text.await_args_list
        for parte, chamada in zip(partes, chamadas):
            self.assertLessEqual(utf16(parte), 4096)
        self.assertTrue(all(not c.kwargs for c in chamadas[:-1]))
        self.assertIn("reply_markup", chamadas[-1].kwargs)
        self.assertIn("Análise da linguagem", partes[0])
        self.assertIn("📌 Alegação 40", partes[-1])
        texto = "\n".join(partes)
        posicoes = [texto.index(f"📌 Alegação {i}\n") for i in range(1, 41)]
        self.assertEqual(posicoes, sorted(posicoes))
        self.assertEqual(
            "".join(texto.split()), "".join(self.bot._formatar_relatorio(self.result).split()),
        )

    async def test_button_starts_dialogue_from_stored_investigation(self):
        self.bot.socratico = MagicMock()
        self.bot.socratico.tem_sessao_ativa.return_value = False
        self.bot.socratico.iniciar_sessao.return_value = "Pergunta 1/3: Qual é a fonte?"
        self.bot._fetch = lambda _id: self.result
        query = SimpleNamespace(
            data="socratico:test-id",
            answer=AsyncMock(),
            from_user=SimpleNamespace(id=7),
            message=SimpleNamespace(
                chat=SimpleNamespace(id=11, type="private"), reply_text=AsyncMock(),
            ),
        )
        await self.bot._iniciar_dialogo_socratico(
            SimpleNamespace(callback_query=query), SimpleNamespace(),
        )
        query.answer.assert_awaited_once_with("Preparando o diálogo…")
        self.bot.socratico.iniciar_sessao.assert_called_once_with(
            7, self.result.texto_original, self.result.vereditos, chat_id=11,
        )
        query.message.reply_text.assert_awaited_once_with(
            "Pergunta 1/3: Qual é a fonte?"
        )

    async def test_private_answer_advances_active_dialogue_without_new_analysis(self):
        self.bot.socratico = MagicMock()
        self.bot.socratico.tem_sessao_ativa.return_value = True
        self.bot.socratico.processar_resposta.return_value = "Pergunta 2/3: Por quê?"
        self.message.from_user = SimpleNamespace(id=7)
        self.message.chat.id = 11
        await self.bot._handle_private_message(self.update, SimpleNamespace())
        self.bot.motor.analisar_texto.assert_not_awaited()
        self.bot.socratico.processar_resposta.assert_called_once_with(
            7, "Texto de teste", 11,
        )
        self.assertEqual(self.message.reply_text.await_count, 2)
        self.assertIn("relacionando sua resposta", self.message.reply_text.await_args_list[0].args[0])
        self.assertEqual(self.message.reply_text.await_args_list[1].args[0], "Pergunta 2/3: Por quê?")


EV_ID = "ev_" + "a" * 64


def veredito(i, categoria, justificativa, fontes):
    return VereditoAlegacao(i, f"Alegação número {i}", categoria, justificativa, fontes)


def utf16(texto):
    return len(texto.encode("utf-16-le")) // 2


class TestReportFormat(unittest.TestCase):
    def setUp(self):
        self.result = ResultadoAnalise(
            analise_linguistica="Score experimental de associação à classe fake: 0,930.",
            vereditos=[
                veredito(
                    1, "refutada",
                    f'[{EV_ID}] Agência: "Não procede" (refuta); https://checagem.example/1',
                    ["https://checagem.example/1"],
                ),
                veredito(2, "evidencia_insuficiente", "Nenhuma evidência encontrada.", []),
            ],
        )
        self.report = BotTelegram._formatar_relatorio(self.result)
        self.factual_start = self.report.index("Verificação factual por alegação")

    def test_sections_are_ordered_and_language_disclaimer_is_explicit(self):
        language_start = self.report.index("Análise da linguagem")
        self.assertLess(language_start, self.factual_start)
        disclaimer = self.report.index("não avalia se as informações são verdadeiras")
        self.assertLess(language_start, disclaimer)
        self.assertLess(disclaimer, self.factual_start)

    def test_language_risk_is_not_mixed_with_factual_verdict(self):
        language, factual = self.report[:self.factual_start], self.report[self.factual_start:]
        self.assertIn(self.result.analise_linguistica, language)
        self.assertNotIn(self.result.analise_linguistica, factual)
        self.assertNotIn("Resultado factual:", language)
        for termo in ("Score", "enganos", "fake"):
            self.assertNotIn(termo, factual)

    def test_claim_block_has_verdict_justification_plain_urls_and_limitations(self):
        self.assertIn('"Alegação número 1"\nResultado factual: refutada', self.report)
        self.assertIn('Agência: "Não procede" (refuta)', self.report)
        self.assertNotIn(EV_ID, self.report)
        self.assertIn("Fontes:\nhttps://checagem.example/1", self.report)
        self.assertIn("Resultado factual: evidência insuficiente", self.report)
        self.assertIn("Nenhuma fonte disponível para esta alegação.", self.report)
        bloco_1, bloco_2 = self.report.split("📌 Alegação ")[1:]
        self.assertNotIn("Recomendamos revisão manual", bloco_1)
        self.assertIn("Limitações:", bloco_2)
        self.assertIn("Recomendamos revisão manual", bloco_2)

    def test_manual_review_for_unverified_and_conflicting(self):
        for categoria in ("nao_verificada", "conflitante"):
            report = BotTelegram._formatar_relatorio(
                ResultadoAnalise(vereditos=[veredito(1, categoria, "Motivo.", [])])
            )
            self.assertIn("Recomendamos revisão manual", report)
        report = BotTelegram._formatar_relatorio(
            ResultadoAnalise(vereditos=[veredito(1, "apoiada", "Motivo.", ["https://a.example"])])
        )
        self.assertNotIn("Limitações:", report)

    def test_unavailable_steps_are_visible(self):
        self.result.etapas_indisponiveis = ["Fonte indisponível: a busca no Tavily não respondeu."]
        report = BotTelegram._formatar_relatorio(self.result)
        self.assertIn("Etapas indisponíveis", report)
        self.assertIn("Fonte indisponível: a busca no Tavily não respondeu.", report)
        self.assertNotIn("Etapas indisponíveis", self.report)

    def test_empty_result_still_has_both_sections(self):
        report = BotTelegram._formatar_relatorio(ResultadoAnalise())
        self.assertIn("Análise linguística indisponível.", report)
        self.assertIn("Nenhuma alegação verificável foi identificada no texto.", report)


class TestDividirMensagem(unittest.TestCase):
    def test_short_and_empty_text(self):
        self.assertEqual(BotTelegram._dividir_mensagem("curto"), ["curto"])
        self.assertEqual(BotTelegram._dividir_mensagem(""), [])
        self.assertEqual(BotTelegram._dividir_mensagem("\n\n \n"), [])

    def test_prefers_paragraph_boundaries(self):
        texto = "a" * 6 + "\n\n" + "b" * 6 + "\n\n" + "c" * 6
        self.assertEqual(BotTelegram._dividir_mensagem(texto, 14), ["a" * 6 + "\n\n" + "b" * 6, "c" * 6])

    def test_oversized_line_is_hard_cut(self):
        partes = BotTelegram._dividir_mensagem("x" * 10000)
        self.assertEqual([len(p) for p in partes], [4096, 4096, 1808])

    def test_emoji_counts_as_two_utf16_units(self):
        partes = BotTelegram._dividir_mensagem("🔎" * 3000)
        self.assertTrue(all(utf16(p) <= 4096 for p in partes))
        self.assertEqual("".join(partes), "🔎" * 3000)


class TestLogging(unittest.TestCase):
    def test_http_request_info_logs_disabled(self):
        names = ("httpx", "httpx2", "httpcore")
        previous = {name: logging.getLogger(name).level for name in names}
        self.addCleanup(lambda: [logging.getLogger(name).setLevel(level) for name, level in previous.items()])
        _configurar_logging()
        for name in names:
            self.assertFalse(logging.getLogger(name).isEnabledFor(logging.INFO))
            self.assertTrue(logging.getLogger(name).isEnabledFor(logging.WARNING))
