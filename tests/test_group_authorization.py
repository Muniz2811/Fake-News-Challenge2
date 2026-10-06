"""MVP-03/04: grupos explícitos, logs seguros e deep links, sem Telegram real."""
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_search  # Isola configuração e serviços externos.
from bot import BotTelegram
from config import parse_authorized_groups
from motor_ia import ResultadoAnalise

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
