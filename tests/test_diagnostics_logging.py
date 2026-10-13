"""Testes de correlação, duração e privacidade da instrumentação."""
import asyncio
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import test_search  # Isola configuração e chamadas externas; não lê .env.
from bot import BotTelegram
from claims import Alegacao, ResultadoExtracao
from diagnostics import (
    current_correlation_id, install_correlation_record_factory,
    investigation_context, logged_async_stage, log_stage,
)
from motor_ia import MotorIA
from search import Evidencia
from verifier import VereditoAlegacao


class TestDiagnosticsLogging(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("diagnostics_test")

    def test_stage_logs_correlation_and_elapsed_time(self):
        with self.assertLogs(self.logger, level="INFO") as captured:
            with investigation_context() as correlation_id:
                with patch("diagnostics.time.perf_counter", side_effect=[10.0, 10.25]):
                    with log_stage(self.logger, "unit_stage"):
                        pass
        self.assertEqual(uuid.UUID(correlation_id).version, 4)
        records = captured.records
        self.assertEqual(len(records), 2)
        self.assertIn("event=started", records[0].getMessage())
        self.assertIn("event=completed", records[1].getMessage())
        self.assertTrue(all(r.correlation_id == correlation_id for r in records))
        self.assertIn("duration_seconds=0.250000", records[1].getMessage())
        self.assertIn("duration_ms=250.000", records[1].getMessage())

    def test_stage_failure_is_categorized_without_exception_payload(self):
        sensitive = "mensagem privada e identificador pessoal"
        with self.assertLogs(self.logger, level="INFO") as captured:
            with investigation_context():
                with self.assertRaises(RuntimeError):
                    with log_stage(self.logger, "failing_stage"):
                        raise RuntimeError(sensitive)
        output = " ".join(captured.output)
        self.assertIn("error_category=INTERNAL_ERROR", output)
        self.assertIn("event=failed", output)
        self.assertIn("duration_ms=", output)
        self.assertIn("RuntimeError", output)
        self.assertNotIn("event=completed", output)
        self.assertNotIn(sensitive, output)
        self.assertEqual(current_correlation_id(), "-")

    def test_external_service_failure_has_distinct_category(self):
        with self.assertLogs(self.logger, level="ERROR") as captured:
            with investigation_context():
                with self.assertRaises(TimeoutError):
                    with log_stage(self.logger, "remote_call"):
                        raise TimeoutError("private payload")
        self.assertIn("error_category=EXTERNAL_SERVICE_UNAVAILABLE", captured.output[0])
        self.assertNotIn("private payload", captured.output[0])

    def test_missing_configuration_has_distinct_category(self):
        with self.assertLogs(self.logger, level="ERROR") as captured:
            with self.assertRaises(ValueError):
                with log_stage(self.logger, "configuration"):
                    raise ValueError("API key ausente: segredo-ficticio")
        self.assertIn("error_category=CONFIGURATION_FAILURE", captured.output[0])
        self.assertNotIn("segredo-ficticio", captured.output[0])

    def test_nested_context_restores_outer_id_and_resets_afterwards(self):
        with investigation_context() as outer:
            with investigation_context() as inherited:
                self.assertEqual(inherited, outer)
            with investigation_context(str(uuid.uuid4())) as inner:
                self.assertNotEqual(inner, outer)
            self.assertEqual(current_correlation_id(), outer)
        self.assertEqual(current_correlation_id(), "-")

    def test_personal_identifier_is_rejected_as_correlation_id(self):
        for invalid in ("usuario@example.org", "123456", str(uuid.uuid1())):
            with self.subTest(identifier=invalid), self.assertRaises(ValueError):
                with investigation_context(invalid):
                    self.fail("Identificador pessoal aceito")
        self.assertEqual(current_correlation_id(), "-")

    def test_record_factory_is_idempotent_and_preserves_previous_factory(self):
        previous = logging.getLogRecordFactory()

        def third_party_factory(*args, **kwargs):
            record = logging.LogRecord(*args, **kwargs)
            record.third_party_field = True
            return record

        try:
            logging.setLogRecordFactory(third_party_factory)
            install_correlation_record_factory()
            installed = logging.getLogRecordFactory()
            install_correlation_record_factory()
            self.assertIs(logging.getLogRecordFactory(), installed)
            with self.assertLogs(self.logger) as captured:
                with investigation_context() as correlation_id:
                    self.logger.info("evento tecnico")
            self.assertTrue(captured.records[0].third_party_field)
            self.assertEqual(captured.records[0].correlation_id, correlation_id)
        finally:
            logging.setLogRecordFactory(previous)


class TestAsyncDiagnostics(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_investigations_and_worker_threads_are_isolated(self):
        logger = logging.getLogger("diagnostics_concurrent_test")

        async def investigate():
            with investigation_context() as identifier:
                await asyncio.sleep(0)

                def worker():
                    logger.info("evento da thread")
                    return current_correlation_id()

                thread_id = await asyncio.to_thread(worker)
                self.assertEqual(current_correlation_id(), identifier)
                return identifier, thread_id

        with self.assertLogs(logger, level="INFO") as captured:
            first, second = await asyncio.gather(investigate(), investigate())
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[0], first[1])
        self.assertEqual(second[0], second[1])
        self.assertEqual({r.correlation_id for r in captured.records}, {first[0], second[0]})
        self.assertEqual(current_correlation_id(), "-")

    async def test_decorated_failure_retains_id_and_restores_context(self):
        @logged_async_stage("test_pipeline")
        async def fail():
            raise RuntimeError("conteudo privado")

        with self.assertLogs(__name__, level="INFO") as captured:
            with self.assertRaises(RuntimeError):
                await fail()
        self.assertEqual(len({r.correlation_id for r in captured.records}), 1)
        self.assertNotEqual(captured.records[0].correlation_id, "-")
        self.assertEqual(current_correlation_id(), "-")

    def make_motor(self, providers):
        analyzer = MagicMock(indisponibilidade=None)
        analyzer.analisar.return_value = 0.9
        extractor = MagicMock()
        extractor.extrair.return_value = ResultadoExtracao("Texto", [Alegacao(1, "A universidade publicou um estudo")])
        verifier = MagicMock()
        verifier.verificar_alegacao.return_value = VereditoAlegacao(
            1, "A universidade publicou um estudo", "apoiada", "Evidência utilizada",
            ["https://example.org/estudo"],
        )
        return MotorIA(providers, verifier, analyzer, llm=MagicMock(), extrator_claims=extractor)

    async def test_bot_and_motor_log_pipeline_only_once_and_preserve_result(self):
        provider = MagicMock()

        def search(_claim):
            logging.getLogger("search").info("evento tecnico da thread")
            return [Evidencia("Estudo", "https://example.org/estudo", "Universidade", "Trecho")]

        provider.buscar_evidencias_por_alegacao.side_effect = search
        bot = BotTelegram.__new__(BotTelegram)
        bot.motor = self.make_motor([provider])
        bot._store = MagicMock(return_value="investigacao")
        bot._enviar_relatorio = AsyncMock()
        message = SimpleNamespace(text="A universidade publicou um estudo", reply_text=AsyncMock())
        with self.assertLogs(level="INFO") as captured:
            await bot._handle_private_message(SimpleNamespace(message=message), SimpleNamespace())
        pipelines = [r for r in captured.records if "stage=analysis_pipeline " in r.getMessage()]
        self.assertEqual(len(pipelines), 2)
        self.assertIn("event=started", pipelines[0].getMessage())
        self.assertIn("event=completed", pipelines[1].getMessage())
        self.assertEqual(len({r.correlation_id for r in captured.records}), 1)
        result = bot._enviar_relatorio.await_args.args[1]
        self.assertEqual(result.score_linguistico, 0.9)
        self.assertEqual(result.vereditos[0].categoria, "apoiada")
        self.assertEqual(result.evidencias[0].url, "https://example.org/estudo")

    async def test_failed_provider_keeps_same_id_and_other_provider_result(self):
        failed = MagicMock()
        failed.buscar_evidencias_por_alegacao.side_effect = RuntimeError("conteudo privado")
        working = MagicMock()
        working.buscar_evidencias_por_alegacao.return_value = [
            Evidencia("Estudo", "https://example.org/estudo", "Universidade", "Trecho"),
        ]
        with self.assertLogs("motor_ia", level="INFO") as captured:
            result = await self.make_motor([failed, working]).analisar_texto("A universidade publicou um estudo")
        failures = [r for r in captured.records if "event=failed" in r.getMessage()]
        self.assertEqual(len(failures), 1)
        self.assertIn("error_category=EXTERNAL_SERVICE_UNAVAILABLE", failures[0].getMessage())
        self.assertEqual(len({r.correlation_id for r in captured.records}), 1)
        self.assertNotIn("conteudo privado", " ".join(captured.output))
        self.assertEqual(result.vereditos[0].categoria, "apoiada")
        self.assertEqual(len(result.evidencias), 1)
        self.assertTrue(result.requer_revisao)

    async def test_bot_error_log_keeps_investigation_id_and_omits_payload(self):
        bot = BotTelegram.__new__(BotTelegram)
        identifiers = []

        async def fail(_text):
            identifiers.append(current_correlation_id())
            raise RuntimeError("mensagem privada")

        bot.motor = SimpleNamespace(analisar_texto=fail)
        message = SimpleNamespace(text="mensagem privada", reply_text=AsyncMock())
        with self.assertLogs("bot", level="ERROR") as captured:
            await bot._handle_private_message(SimpleNamespace(message=message), SimpleNamespace())
        self.assertEqual(captured.records[0].correlation_id, identifiers[0])
        self.assertNotEqual(identifiers[0], "-")
        self.assertNotIn("mensagem privada", " ".join(captured.output))
        self.assertEqual(current_correlation_id(), "-")


if __name__ == "__main__":
    unittest.main()
