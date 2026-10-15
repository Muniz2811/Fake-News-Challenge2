"""Preserva baseline, múltiplos provedores e revisão da MVP-13 sem APIs reais."""
import unittest
from unittest.mock import MagicMock, patch

import test_search
import requests
from claims import Alegacao, ResultadoExtracao
from motor_ia import MotorIA
from search import Evidencia, GoogleFactCheckAdapter, TavilySearchAdapter
from verifier import VereditoAlegacao, VerificadorAlegacoes

CLAIM = "A universidade publicou o relatório em outubro"
SECRET_TEXT = "Conteúdo da mensagem que nunca deve ser registrado"


class TestMotorResilience(unittest.IsolatedAsyncioTestCase):
    def make_motor(self, providers, verifier=None):
        analyzer = MagicMock()
        analyzer.analisar.return_value = 0.42
        analyzer.indisponibilidade = None
        motor = MotorIA(providers, verifier or VerificadorAlegacoes(), analyzer)
        motor.extrator_claims = MagicMock()
        motor.extrator_claims.extrair.return_value = ResultadoExtracao(CLAIM, [Alegacao(1, CLAIM)])
        return motor

    async def test_adapter_timeout_is_visible_and_other_provider_still_works(self):
        failed = TavilySearchAdapter("test-key")
        working = MagicMock()
        working.buscar_evidencias_por_alegacao.return_value = [Evidencia(
            title="Checagem", url="https://example.org/checagem", publisher="Fonte",
            snippet="Veredito do Checador: Verdadeiro", checked_claim=CLAIM,
            textual_rating="Verdadeiro", source_type="fact_check",
        )]
        motor = self.make_motor([failed, working])
        with patch("search.requests.post", side_effect=requests.exceptions.Timeout(SECRET_TEXT)):
            with self.assertLogs("search", level="WARNING") as captured:
                result = await motor.analisar_texto(CLAIM)
        self.assertNotIn(SECRET_TEXT, " ".join(captured.output))
        self.assertEqual(result.vereditos[0].categoria, "apoiada")
        self.assertEqual(result.score_linguistico, 0.42)
        self.assertTrue(result.requer_revisao)
        self.assertEqual(result.etapas_indisponiveis, ["Fonte indisponível: a busca no Tavily não respondeu."])
        working.buscar_evidencias_por_alegacao.assert_called_once_with(CLAIM)
        self.assertTrue(hasattr(motor, "socratico"))

    async def test_empty_success_is_not_reported_as_provider_outage(self):
        provider = TavilySearchAdapter("test-key")
        provider.ultimo_erro = "Timeout"
        response = MagicMock(status_code=200)
        response.json.return_value = {"results": []}
        with patch("search.requests.post", return_value=response):
            result = await self.make_motor([provider]).analisar_texto(CLAIM)
        self.assertIsNone(provider.ultimo_erro)
        self.assertEqual(result.etapas_indisponiveis, [])
        self.assertTrue(result.requer_revisao)
        self.assertEqual(result.vereditos[0].categoria, "evidencia_insuficiente")

    async def test_language_only_does_not_extract_claims_search_evidence_or_call_verifier(self):
        provider = MagicMock()
        verifier = MagicMock()
        motor = self.make_motor([provider], verifier)
        result = await motor.analisar_linguagem(CLAIM)
        self.assertEqual(result.score_linguistico, .42)
        self.assertEqual(result.texto_original, CLAIM)
        self.assertEqual(result.alegacoes, [])
        self.assertEqual(result.evidencias, [])
        self.assertEqual(result.vereditos, [])
        self.assertEqual(result.resultado_factual, "")
        motor.extrator_claims.extrair.assert_not_called()
        provider.buscar_evidencias_por_alegacao.assert_not_called()
        verifier.verificar_alegacao.assert_not_called()

    async def test_high_language_score_does_not_override_supported_claim(self):
        verifier = MagicMock()
        verifier.verificar_alegacao.return_value = VereditoAlegacao(
            1, CLAIM, "apoiada", "Evidência comparável usada", ["https://example.org/estudo"],
        )
        motor = self.make_motor([], verifier)
        motor.analisador_linguistico.analisar.return_value = 0.99
        result = await motor.analisar_texto(CLAIM)
        self.assertEqual(result.score_linguistico, 0.99)
        self.assertEqual(result.vereditos[0].categoria, "apoiada")
        self.assertFalse(result.requer_revisao)

    async def test_extraction_failure_keeps_score_and_reports_missing_stage(self):
        motor = self.make_motor([])
        motor.extrator_claims.extrair.side_effect = RuntimeError(SECRET_TEXT)
        with self.assertLogs("motor_ia", level="ERROR") as captured:
            result = await motor.analisar_texto(CLAIM)
        self.assertEqual(result.score_linguistico, 0.42)
        self.assertTrue(result.requer_revisao)
        self.assertIn("Extração de alegações indisponível", result.etapas_indisponiveis[0])
        self.assertNotIn(SECRET_TEXT, " ".join(captured.output))

    async def test_verification_failure_does_not_erase_other_claims(self):
        verifier = MagicMock()
        verifier.verificar_alegacao.side_effect = [
            RuntimeError(SECRET_TEXT), VereditoAlegacao(2, "Outra alegação", "apoiada", "Fonte usada"),
        ]
        motor = self.make_motor([], verifier)
        motor.extrator_claims.extrair.return_value.alegacoes.append(Alegacao(2, "Outra alegação"))
        with self.assertLogs("motor_ia", level="ERROR") as captured:
            result = await motor.analisar_texto(CLAIM)
        self.assertEqual([v.categoria for v in result.vereditos], ["nao_verificada", "apoiada"])
        self.assertEqual(result.score_linguistico, 0.42)
        self.assertNotIn(SECRET_TEXT, " ".join(captured.output))

    async def test_language_failure_does_not_block_factual_verification(self):
        motor = self.make_motor([])
        motor.analisador_linguistico.analisar.side_effect = RuntimeError(SECRET_TEXT)
        with self.assertLogs("motor_ia", level="ERROR") as captured:
            result = await motor.analisar_texto(CLAIM)
        self.assertIsNone(result.score_linguistico)
        self.assertEqual(len(result.vereditos), 1)
        self.assertTrue(result.requer_revisao)
        self.assertNotIn(SECRET_TEXT, " ".join(captured.output))

    async def test_google_http_failure_is_visible_without_logging_query(self):
        response = MagicMock(status_code=429)
        with patch("search.requests.get", return_value=response):
            with self.assertLogs("search", level="WARNING") as captured:
                result = await self.make_motor([GoogleFactCheckAdapter("test-key")]).analisar_texto(CLAIM)
        self.assertIn("Google Fact Check", result.etapas_indisponiveis[0])
        self.assertNotIn(CLAIM, " ".join(captured.output))
