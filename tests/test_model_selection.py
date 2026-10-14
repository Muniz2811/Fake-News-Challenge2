"""Escolha de modelo por usuário e investigação; Telegram e modelos simulados."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import test_search  # Não lê .env e não chama serviços reais.
from bot import BotTelegram
from claims import Alegacao, ResultadoExtracao
from model_catalog import MODEL_NAMES, ModelCatalog
from motor_ia import MotorIA, ResultadoAnalise
from telegram.error import BadRequest
from verifier import VereditoAlegacao


def analyzer(score, name):
    return SimpleNamespace(indisponibilidade=None, descricao_modelo=name, analisar=MagicMock(return_value=score))


class TestModelCatalog(unittest.TestCase):
    def test_models_are_loaded_on_demand_and_reused(self):
        model = analyzer(0.2, "BERTimbau")
        factory = MagicMock(return_value=model)
        catalog = ModelCatalog({"bertimbau_mlp": factory})
        factory.assert_not_called()
        self.assertIs(catalog.get("bertimbau_mlp"), model)
        self.assertIs(catalog.get("bertimbau_mlp"), model)
        factory.assert_called_once_with()

    def test_missing_model_can_be_retried_after_artifacts_are_fixed(self):
        missing = SimpleNamespace(indisponibilidade="artefatos ausentes")
        ready = analyzer(0.2, "BERTimbau")
        factory = MagicMock(side_effect=[missing, ready])
        catalog = ModelCatalog({"bertimbau_mlp": factory})
        self.assertIs(catalog.get("bertimbau_mlp"), missing)
        self.assertIs(catalog.get("bertimbau_mlp"), ready)
        self.assertIs(catalog.get("bertimbau_mlp"), ready)
        self.assertEqual(factory.call_count, 2)

    def test_concurrent_requests_only_load_model_once(self):
        model = analyzer(0.2, "BERTimbau")
        factory = MagicMock(return_value=model)
        catalog = ModelCatalog({"bertimbau_mlp": factory})
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(catalog.get, ["bertimbau_mlp"] * 8))
        self.assertTrue(all(result is model for result in results))
        factory.assert_called_once_with()

    def test_encoder_is_shared_only_when_configuration_matches(self):
        catalog = ModelCatalog({})
        directory = Path("models/language_risk/bertimbau_base")
        with patch("language_risk._load_bertimbau_encoder", side_effect=[object(), object()]) as load:
            first = catalog.encoder(directory, {"max_tokens": 256}, "cpu")
            second = catalog.encoder(directory, {"max_tokens": 256}, "cpu")
            different = catalog.encoder(directory, {"max_tokens": 128}, "cpu")
        self.assertIs(first, second)
        self.assertIsNot(first, different)
        self.assertEqual(load.call_count, 2)

    def test_unknown_model_never_calls_factory(self):
        factory = MagicMock()
        catalog = ModelCatalog({"tfidf": factory})
        with self.assertRaises(ValueError):
            catalog.get("../../.env")
        factory.assert_not_called()


class TestMotorModelSelection(unittest.IsolatedAsyncioTestCase):
    def make_motor(self):
        baseline = analyzer(0.15, MODEL_NAMES["tfidf"])
        extractor = MagicMock()
        extractor.extrair.return_value = ResultadoExtracao("Texto", [Alegacao(1, "O estudo foi publicado")])
        verifier = MagicMock()
        verifier.verificar_alegacao.return_value = VereditoAlegacao(
            1, "O estudo foi publicado", "apoiada", "Fonte comparável", ["https://example.org/estudo"],
        )
        with patch("motor_ia.LANGUAGE_RISK_BACKEND", "tfidf"):
            motor = MotorIA([], verifier, baseline, llm=MagicMock(), extrator_claims=extractor)
        return motor, baseline

    async def test_different_requests_do_not_change_global_model_or_factual_verdict(self):
        motor, baseline = self.make_motor()
        mlp = analyzer(0.85, MODEL_NAMES["bertimbau_mlp"])
        with patch("motor_ia.AnalisadorBertimbauMLP", return_value=mlp):
            selected, default = await asyncio.gather(
                motor.analisar_texto("Texto", modelo_id="bertimbau_mlp"),
                motor.analisar_texto("Texto"),
            )
        self.assertEqual(selected.score_linguistico, 0.85)
        self.assertEqual(default.score_linguistico, 0.15)
        self.assertEqual(selected.modelo_linguistico, "bertimbau_mlp")
        self.assertEqual(default.modelo_linguistico, "tfidf")
        self.assertEqual(selected.vereditos, default.vereditos)
        self.assertEqual(selected.resultado_factual, default.resultado_factual)
        self.assertIs(motor.analisador_linguistico, baseline)
        self.assertIn("MLP", selected.analise_linguistica)

    async def test_load_error_keeps_factual_results_and_requested_model_name(self):
        motor, _ = self.make_motor()
        with patch("motor_ia.AnalisadorBertimbauLogReg", side_effect=RuntimeError("conteúdo privado")):
            with self.assertLogs("motor_ia", level="ERROR") as captured:
                result = await motor.analisar_texto("Texto", modelo_id="bertimbau_logreg")
        self.assertIsNone(result.score_linguistico)
        self.assertEqual(result.vereditos[0].categoria, "apoiada")
        self.assertEqual(result.modelo_linguistico, "bertimbau_logreg")
        self.assertIn("BERTimbau", result.analise_linguistica)
        self.assertNotIn("conteúdo privado", " ".join(captured.output))

    async def test_two_bert_classifiers_share_encoder_via_factories(self):
        motor, _ = self.make_motor()
        directory = Path("models/language_risk/bertimbau_base")

        def construct(*args, encoder_loader, **kwargs):
            encoder_loader(directory, {"dimension": 768}, "cpu")
            return analyzer(0.5, "BERTimbau")

        with patch("language_risk._load_bertimbau_encoder") as load:
            with patch("motor_ia.AnalisadorBertimbauLogReg", side_effect=construct), patch(
                "motor_ia.AnalisadorBertimbauMLP", side_effect=construct
            ):
                await asyncio.to_thread(motor.preparar_modelo, "bertimbau_logreg")
                await asyncio.to_thread(motor.preparar_modelo, "bertimbau_mlp")
        load.assert_called_once_with(directory, {"dimension": 768}, "cpu")


class TestTelegramModelSelection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = BotTelegram.__new__(BotTelegram)
        self.result = ResultadoAnalise(analise_linguistica="Modelo de teste")
        self.bot.motor = SimpleNamespace(
            modelo_padrao="tfidf", preparar_modelo=MagicMock(return_value=None),
            analisar_texto=AsyncMock(return_value=self.result),
        )
        self.bot._modelos_por_usuario = {}
        self.bot._modelos_carregando = set()
        self.bot._investigacoes = {}
        self.bot.socratico = MagicMock()
        self.bot.socratico.tem_sessao_ativa.return_value = False
        self.message = SimpleNamespace(
            text="O estudo foi publicado", chat=SimpleNamespace(id=11, type="private"),
            from_user=SimpleNamespace(id=7), reply_text=AsyncMock(),
        )
        self.update = SimpleNamespace(message=self.message)
        self.query = SimpleNamespace(
            message=self.message, from_user=self.message.from_user,
            data="modelo:bertimbau_mlp", answer=AsyncMock(),
        )
        self.callback = SimpleNamespace(callback_query=self.query)

    async def test_command_and_callback_are_registered(self):
        with patch("bot.ApplicationBuilder") as builder:
            instance = BotTelegram(motor=self.bot.motor, token="123456:test-token")
        handlers = [call.args[0] for call in builder.return_value.token.return_value.build.return_value.add_handler.call_args_list]
        self.assertTrue(any("modelo" in getattr(handler, "commands", ()) for handler in handlers))
        self.assertTrue(any(handler.callback == instance._selecionar_modelo for handler in handlers))

    async def test_command_shows_all_choices_without_loading_them(self):
        await self.bot._cmd_modelo(self.update, SimpleNamespace())
        call = self.message.reply_text.await_args
        buttons = call.kwargs["reply_markup"].inline_keyboard
        self.assertEqual([row[0].callback_data for row in buttons], [f"modelo:{key}" for key in MODEL_NAMES])
        self.assertIn("TF-IDF", call.args[0])
        self.bot.motor.preparar_modelo.assert_not_called()

    async def test_selection_applies_to_next_private_message(self):
        await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.assertEqual(self.bot._modelos_por_usuario[(11, 7)], "bertimbau_mlp")
        await self.bot._handle_private_message(self.update, SimpleNamespace())
        self.bot.motor.analisar_texto.assert_awaited_once_with(self.message.text, modelo_id="bertimbau_mlp")

    async def test_selection_is_isolated_by_user_and_chat(self):
        await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.assertEqual(self.bot._modelo_atual(7, 11), "bertimbau_mlp")
        self.assertEqual(self.bot._modelo_atual(8, 11), "tfidf")
        self.assertEqual(self.bot._modelo_atual(7, 12), "tfidf")
        self.message.from_user = SimpleNamespace(id=8)
        await self.bot._handle_private_message(self.update, SimpleNamespace())
        self.bot.motor.analisar_texto.assert_awaited_once_with(self.message.text)

    async def test_unavailable_model_keeps_previous_selection(self):
        self.bot._modelos_por_usuario[(11, 7)] = "bertimbau_logreg"
        self.bot.motor.preparar_modelo.return_value = "dependências BERTimbau não instaladas"
        await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.assertEqual(self.bot._modelos_por_usuario[(11, 7)], "bertimbau_logreg")
        self.assertIn("Modelo indisponível", self.message.reply_text.await_args.args[0])
        self.assertEqual(self.bot._modelos_carregando, set())

    async def test_load_exception_does_not_log_payload_or_change_selection(self):
        self.bot.motor.preparar_modelo.side_effect = RuntimeError("mensagem privada")
        with self.assertLogs("bot", level="ERROR") as captured:
            await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.assertEqual(self.bot._modelos_por_usuario, {})
        self.assertNotIn("mensagem privada", " ".join(captured.output))
        self.assertEqual(self.bot._modelos_carregando, set())

    async def test_forged_callback_and_group_callback_cannot_select_model(self):
        self.query.data = "modelo:../../.env"
        await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.query.data = "modelo:bertimbau_mlp"
        self.message.chat.type = "supergroup"
        await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.bot.motor.preparar_modelo.assert_not_called()
        self.assertEqual(self.bot._modelos_por_usuario, {})

    async def test_expired_callback_does_not_load_or_change_model(self):
        self.query.answer.side_effect = BadRequest("callback expirado")
        with self.assertLogs("bot", level="WARNING"):
            await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.bot.motor.preparar_modelo.assert_not_called()
        self.assertEqual(self.bot._modelos_por_usuario, {})

    async def test_pending_selection_does_not_load_twice(self):
        started, release = Event(), Event()

        def prepare(_model_id):
            started.set()
            if not release.wait(3):
                raise RuntimeError("Teste não liberou o carregamento")

        self.bot.motor.preparar_modelo.side_effect = prepare
        task = asyncio.create_task(self.bot._selecionar_modelo(self.callback, SimpleNamespace()))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
            self.bot.motor.preparar_modelo.assert_called_once_with("bertimbau_mlp")
            self.assertIn("Aguarde", self.message.reply_text.await_args.args[0])
        finally:
            release.set()
            await task

    async def test_analysis_captures_choice_before_acknowledgement(self):
        self.bot._modelos_por_usuario[(11, 7)] = "bertimbau_logreg"

        async def reply(*args, **kwargs):
            self.bot._modelos_por_usuario[(11, 7)] = "bertimbau_mlp"

        self.message.reply_text.side_effect = reply
        await self.bot._handle_private_message(self.update, SimpleNamespace())
        self.bot.motor.analisar_texto.assert_awaited_once_with(self.message.text, modelo_id="bertimbau_logreg")

    async def test_old_deep_link_preserves_original_model_after_selection_changes(self):
        self.result.analise_linguistica = "Modelo local: TF-IDF + regressão logística."
        self.result.modelo_linguistico = "tfidf"
        identifier = self.bot._store(self.result)
        self.bot._modelos_por_usuario[(11, 7)] = "bertimbau_mlp"
        await self.bot._cmd_start(self.update, SimpleNamespace(args=[identifier]))
        self.assertIn("Modelo local: TF-IDF", self.message.reply_text.await_args.args[0])
        self.assertEqual(self.bot._fetch(identifier).modelo_linguistico, "tfidf")

    async def test_group_analysis_uses_default_even_when_author_has_private_choice(self):
        self.bot.authorized_groups = [-1001]
        self.bot.group_analysis_mode = "automatic"
        self.message.chat_id = -1001
        self.message.chat.type = "supergroup"
        self.message.chat.id = -1001
        self.bot._modelos_por_usuario[(-1001, 7)] = "bertimbau_mlp"
        await self.bot._handle_group_message(self.update, SimpleNamespace())
        self.bot.motor.analisar_texto.assert_awaited_once_with(self.message.text)

    async def test_active_dialogue_keeps_its_flow_after_model_choice(self):
        self.bot._modelos_por_usuario[(11, 7)] = "bertimbau_mlp"
        self.bot.socratico.tem_sessao_ativa.return_value = True
        self.bot.socratico.processar_resposta.return_value = "Pergunta seguinte"
        await self.bot._handle_private_message(self.update, SimpleNamespace())
        self.bot.motor.analisar_texto.assert_not_awaited()
        self.bot.socratico.processar_resposta.assert_called_once_with(7, self.message.text, 11)

    async def test_preferences_are_bounded(self):
        self.bot._modelos_por_usuario[(10, 6)] = "tfidf"
        with patch("bot._MAX_ESCOLHAS_MODELO", 1):
            await self.bot._selecionar_modelo(self.callback, SimpleNamespace())
        self.assertEqual(self.bot._modelos_por_usuario, {(11, 7): "bertimbau_mlp"})


if __name__ == "__main__":
    unittest.main()
