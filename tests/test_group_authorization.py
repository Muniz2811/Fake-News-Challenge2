"""MVP-03/04: grupos explícitos, logs seguros e deep links, sem Telegram real."""
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_search  # Isola configuração e serviços externos.
from bot import BotTelegram
from config import parse_authorized_groups
from motor_ia import ResultadoAnalise
from telegram import Chat, Message, MessageEntity, Update
from verifier import VereditoAlegacao

TEXT = "Texto privado de teste que não pode aparecer em logs ou no alerta."


class TestGroupConfiguration(unittest.TestCase):
    def test_empty_configuration_denies_all_groups(self):
        self.assertEqual(parse_authorized_groups(""), [])
        self.assertEqual(parse_authorized_groups(" , "), [])

    def test_multiple_ids_and_duplicates(self):
        self.assertEqual(parse_authorized_groups("-1001, -1002,-1001"), [-1001, -1002])

    def test_invalid_id_has_clear_error_without_echoing_input(self):
        with self.assertRaisesRegex(ValueError, "IDs numéricos") as error:
            parse_authorized_groups("-1001,conteudo-confidencial")
        self.assertNotIn("conteudo-confidencial", str(error.exception))


class TestGroupAuthorization(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.result = ResultadoAnalise(analise_linguistica="Score experimental", requer_revisao=True)
        self.bot = BotTelegram.__new__(BotTelegram)
        self.bot.authorized_groups = [-1001]
        self.bot.group_analysis_mode = "automatic"
        self.bot._investigacoes = {}
        self.bot.motor = SimpleNamespace(analisar_texto=AsyncMock(return_value=self.result))
        self.message = SimpleNamespace(
            text=TEXT, chat_id=-1001, chat=SimpleNamespace(type="supergroup"),
            reply_text=AsyncMock(),
        )
        self.update = SimpleNamespace(message=self.message)
        self.context = SimpleNamespace(bot=SimpleNamespace(
            get_me=AsyncMock(return_value=SimpleNamespace(username="contraponto_test_bot")),
        ))

    async def test_empty_authorization_does_not_process(self):
        self.bot.authorized_groups = []
        await self.bot._handle_group_message(self.update, self.context)
        self.bot.motor.analisar_texto.assert_not_awaited()
        self.message.reply_text.assert_not_awaited()

    async def test_unauthorized_group_is_ignored(self):
        self.message.chat_id = -1002
        await self.bot._handle_group_message(self.update, self.context)
        self.bot.motor.analisar_texto.assert_not_awaited()
        self.context.bot.get_me.assert_not_awaited()

    async def test_nontext_empty_and_private_do_not_enter_group_flow(self):
        for text, kind in [(None, "group"), (" ", "group"), (TEXT, "private")]:
            self.message.text = text
            self.message.chat.type = kind
            await self.bot._handle_group_message(self.update, self.context)
        self.bot.motor.analisar_texto.assert_not_awaited()

    async def test_group_text_is_analyzed_without_alert_when_review_not_required(self):
        self.result.requer_revisao = False
        await self.bot._handle_group_message(self.update, self.context)
        self.bot.motor.analisar_texto.assert_awaited_once_with(TEXT)
        self.message.reply_text.assert_not_awaited()
        self.assertFalse(self.bot._investigacoes)

    async def test_group_alert_deep_link_recovers_private_report(self):
        await self.bot._handle_group_message(self.update, self.context)
        call = self.message.reply_text.await_args
        self.assertNotIn(TEXT, call.args[0])
        self.assertNotIn("fake", call.args[0].casefold())
        self.assertNotIn("Relatório de Investigação", call.args[0])
        link = call.kwargs["reply_markup"].inline_keyboard[0][0].url
        self.assertTrue(link.startswith("https://t.me/contraponto_test_bot?start="))
        identifier = link.split("?start=", 1)[1]
        self.assertIs(self.bot._fetch(identifier), self.result)
        private_message = SimpleNamespace(chat=SimpleNamespace(type="private"), reply_text=AsyncMock())
        await self.bot._cmd_start(SimpleNamespace(message=private_message), SimpleNamespace(args=[identifier]))
        self.assertIn("Relatório de Investigação", private_message.reply_text.await_args.args[0])
        self.assertIn("Análise da linguagem", private_message.reply_text.await_args.args[0])

    async def test_invalid_and_expired_links_receive_orientation(self):
        identifier = self.bot._store(self.result)
        self.bot._investigacoes[identifier]["expires"] = datetime.now() - timedelta(seconds=1)
        self.message.chat.type = "private"
        for value in ["inexistente", identifier]:
            await self.bot._cmd_start(self.update, SimpleNamespace(args=[value]))
            self.assertIn("não é mais válido ou expirou", self.message.reply_text.await_args.args[0])
        self.assertNotIn(identifier, self.bot._investigacoes)

    async def test_chatid_only_returns_current_group_metadata(self):
        await self.bot._cmd_chatid(self.update, self.context)
        self.message.reply_text.assert_awaited_once_with("ID deste grupo: -1001")
        self.bot.motor.analisar_texto.assert_not_awaited()

    async def test_errors_do_not_log_content_or_group_id(self):
        self.bot.motor.analisar_texto.side_effect = RuntimeError(TEXT)
        with self.assertLogs("bot", level="ERROR") as captured:
            await self.bot._handle_group_message(self.update, self.context)
            await self.bot._handle_error(self.update, SimpleNamespace(error=ValueError(TEXT)))
        output = " ".join(captured.output)
        self.assertNotIn(TEXT, output)
        self.assertNotIn("-1001", output)
        self.assertIn("RuntimeError", output)

    def test_explicit_empty_list_overrides_configured_groups(self):
        with patch("bot.ApplicationBuilder"), patch("bot.AUTHORIZED_GROUPS", [-1001]):
            default_bot = BotTelegram(self.bot.motor, "123456:fake-token")
            private_only_bot = BotTelegram(self.bot.motor, "123456:fake-token", [])
        self.assertEqual(default_bot.authorized_groups, [-1001])
        self.assertEqual(private_only_bot.authorized_groups, [])


class TestGroupCommand(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.result = ResultadoAnalise(
            analise_linguistica="Modelo local: TF-IDF + regressão logística.",
            requer_revisao=False,
            vereditos=[VereditoAlegacao(
                1, TEXT, "evidencia_insuficiente", "Faltam dados comparáveis.",
                ["https://example.org/estudo"],
            )],
        )
        self.bot = BotTelegram.__new__(BotTelegram)
        self.bot.authorized_groups = [-1001]
        self.bot.group_analysis_mode = "command"
        self.bot._investigacoes = {}
        self.bot._modelos_por_usuario = {(11, 7): "bertimbau_mlp"}
        self.message = SimpleNamespace(
            text="/analisar", chat_id=-1001, chat=SimpleNamespace(type="supergroup"),
            reply_to_message=SimpleNamespace(text=TEXT), reply_text=AsyncMock(),
        )
        self.update = SimpleNamespace(message=self.message)
        self.context = SimpleNamespace()

        async def analyze(text):
            # A confirmação deve chegar antes do processamento demorado.
            self.assertEqual(self.message.reply_text.await_count, 1)
            self.assertIn("investigação começou", self.message.reply_text.await_args.args[0])
            return self.result

        self.bot.motor = SimpleNamespace(analisar_texto=AsyncMock(side_effect=analyze))

    async def test_command_analyzes_replied_text_with_default_and_sends_full_report(self):
        await self.bot._cmd_analisar(self.update, self.context)
        self.bot.motor.analisar_texto.assert_awaited_once_with(TEXT)
        calls = self.message.reply_text.await_args_list
        self.assertEqual(len(calls), 2)
        report = calls[1].args[0]
        self.assertIn("Relatório de Investigação", report)
        self.assertIn("TF-IDF", report)
        self.assertIn(TEXT, report)
        self.assertIn("evidência insuficiente", report)
        self.assertIn("Faltam dados comparáveis", report)
        self.assertIn("https://example.org/estudo", report)
        self.assertNotIn("reply_markup", calls[1].kwargs)
        self.assertNotIn("parse_mode", calls[1].kwargs)
        self.assertEqual(self.bot._investigacoes, {})

    async def test_plain_group_text_is_ignored_in_command_mode(self):
        self.message.text = TEXT
        await self.bot._handle_group_message(self.update, self.context)
        self.bot.motor.analisar_texto.assert_not_awaited()
        self.message.reply_text.assert_not_awaited()

    async def test_empty_or_unauthorized_group_cannot_request_analysis(self):
        for groups in ([], [-1002]):
            self.bot.authorized_groups = groups
            await self.bot._cmd_analisar(self.update, self.context)
            self.assertIn("não está autorizado", self.message.reply_text.await_args.args[0])
        self.bot.motor.analisar_texto.assert_not_awaited()

    async def test_missing_reply_nontext_and_empty_original_receive_orientation(self):
        for reply in (None, SimpleNamespace(), SimpleNamespace(text=None, caption=TEXT),
                      SimpleNamespace(text="  ")):
            self.message.reply_to_message = reply
            await self.bot._cmd_analisar(self.update, self.context)
            self.assertIn("Responda a uma mensagem de texto", self.message.reply_text.await_args.args[0])
        self.bot.motor.analisar_texto.assert_not_awaited()

    async def test_private_command_receives_orientation_without_analyzing(self):
        self.message.chat.type = "private"
        await self.bot._cmd_analisar(self.update, self.context)
        self.assertIn("No privado, envie o texto diretamente", self.message.reply_text.await_args.args[0])
        self.bot.motor.analisar_texto.assert_not_awaited()

    async def test_failure_notifies_group_without_logging_content(self):
        self.bot.motor.analisar_texto.side_effect = RuntimeError(TEXT)
        with self.assertLogs("bot", level="ERROR") as captured:
            await self.bot._cmd_analisar(self.update, self.context)
        self.assertEqual(self.message.reply_text.await_count, 2)
        self.assertIn("Não consegui concluir", self.message.reply_text.await_args.args[0])
        self.assertNotIn(TEXT, " ".join(captured.output))
        self.assertNotIn("-1001", " ".join(captured.output))
        self.assertFalse(self.bot._investigacoes)

    async def test_long_group_report_is_split_without_private_dialogue_buttons(self):
        self.result.vereditos[0].justificativa = "Contexto da evidência. 🔎 " * 600
        await self.bot._cmd_analisar(self.update, self.context)
        calls = self.message.reply_text.await_args_list[1:]
        self.assertGreater(len(calls), 1)
        for call in calls:
            self.assertLessEqual(len(call.args[0].encode("utf-16-le")) // 2, 4096)
            self.assertNotIn("reply_markup", call.kwargs)
        combined = "".join(call.args[0] for call in calls)
        self.assertEqual("".join(combined.split()), "".join(self.bot._formatar_relatorio(self.result).split()))

    async def test_explicit_command_still_works_in_automatic_mode(self):
        self.bot.group_analysis_mode = "automatic"
        await self.bot._cmd_analisar(self.update, self.context)
        self.bot.motor.analisar_texto.assert_awaited_once_with(TEXT)

    def test_registered_command_routes_addressed_messages_to_this_bot_only(self):
        with patch("bot.ApplicationBuilder") as builder, patch("bot.GROUP_ANALYSIS_MODE", "command"):
            instance = BotTelegram(motor=self.bot.motor, token="123456:test-token")
        self.assertEqual(instance.group_analysis_mode, "command")
        handlers = [call.args[0] for call in builder.return_value.token.return_value.build.return_value.add_handler.call_args_list]
        handler = next(h for h in handlers if "analisar" in getattr(h, "commands", ()))
        self.assertEqual(handler.callback, instance._cmd_analisar)
        for username, expected in (("contraponto_test_bot", True), ("outro_bot", False)):
            command = f"/analisar@{username}"
            message = Message(
                message_id=2, date=datetime.now(), chat=Chat(-1001, "supergroup"), text=command,
                entities=[MessageEntity("bot_command", 0, len(command))],
            )
            message.set_bot(SimpleNamespace(username="contraponto_test_bot"))
            self.assertEqual(bool(handler.check_update(Update(1, message=message))), expected)

    def test_constructor_rejects_invalid_mode_and_accepts_automatic(self):
        with patch("bot.ApplicationBuilder"):
            instance = BotTelegram(self.bot.motor, "123456:test-token", group_analysis_mode="automatic")
            self.assertEqual(instance.group_analysis_mode, "automatic")
            with self.assertRaisesRegex(ValueError, "GROUP_ANALYSIS_MODE"):
                BotTelegram(self.bot.motor, "123456:test-token", group_analysis_mode="invalid")
