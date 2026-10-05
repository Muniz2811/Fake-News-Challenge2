"""Interface do Contraponto com o Telegram (python-telegram-bot ≥ 20)."""

import os
import sys
import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Adiciona o diretório atual ao sys.path para permitir a importação direta dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TELEGRAM_BOT_TOKEN
from motor_ia import MotorIA, ResultadoAnalise


logger = logging.getLogger(__name__)


class Config:
    """Configurações específicas do bot, lidas do ambiente."""

    def __init__(self) -> None:
        self.telegram_bot_token = TELEGRAM_BOT_TOKEN
        raw_groups = os.getenv("AUTHORIZED_GROUPS", "")
        try:
            self.authorized_groups = {
                int(group_id.strip())
                for group_id in raw_groups.split(",")
                if group_id.strip()
            }
        except ValueError as exc:
            raise ValueError(
                "AUTHORIZED_GROUPS deve conter IDs numéricos separados por vírgula."
            ) from exc


class BotTelegram:

    def __init__(self, motor: Optional[MotorIA] = None) -> None:
        self.config = Config()
        self.motor = motor or MotorIA()

        token = self.config.telegram_bot_token
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN não configurado no arquivo .env.")
        self.app = ApplicationBuilder().token(token).build()

        # /start com deep-link (chat privado)
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("ajuda", self._cmd_ajuda))

        # Mensagens de texto em grupos autorizados
        self.app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
                self._handle_group_message,
            )
        )
        self.app.add_handler(
            MessageHandler(
                filters.ChatType.PRIVATE & ~filters.TEXT,
                self._handle_incompatible_content,
            )
        )
        self.app.add_error_handler(self._handle_error)

        # Mensagens de texto no privado (análise avulsa)
        self.app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
                self._handle_private_message,
            )
        )

    # ------------------------------------------------------------------ #
    #  Handlers                                                           #
    # ------------------------------------------------------------------ #

    async def _handle_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Encaminha texto apenas de grupos autorizados ao motor, sem o reter."""
        if not update.message or not update.message.text:
            return

        chat_id = update.message.chat_id
        if chat_id not in self.config.authorized_groups:
            return

        try:
            await self.motor.analisar_texto(update.message.text)
        except NotImplementedError:
            logger.debug("O motor ainda não está implementado.")
            return
        except Exception as exc:
            # Exceções podem incluir o texto recebido; registre apenas o tipo.
            logger.error(
                "Falha ao encaminhar texto ao motor (%s).",
                type(exc).__name__,
            )

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Trata /start (com ou sem payload de deep-link) no chat privado."""
        if not update.message or update.message.chat.type != "private":
            return

        if not context.args:
            await update.message.reply_text(
                "Olá! Envie uma mensagem de texto diretamente e eu farei uma análise."
            )
            return

        await update.message.reply_text(
            "Envie mensagens textuais em um grupo autorizado para encaminhá-las ao motor."
        )

    async def _cmd_ajuda(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or update.message.chat.type != "private":
            return
        await update.message.reply_text(
            "Envie uma notícia ou alegação como texto para iniciar uma investigação. "
            "Áudios, figurinhas e arquivos ainda não são aceitos. A análise atual é simulada."
        )

    async def _handle_incompatible_content(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message:
            await update.message.reply_text(
                "Por enquanto consigo investigar apenas texto. "
                "Envie a notícia ou alegação como uma mensagem escrita."
            )

    async def _handle_private_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Permite análise avulsa no chat privado."""
        if not update.message:
            return

        texto = update.message.text or ""
        if not texto.strip():
            await update.message.reply_text(
                "Não recebi um texto para investigar. Envie a notícia ou alegação por escrito."
            )
            return

        try:
            # A confirmação é enviada antes de chamar o motor, que pode demorar.
            await update.message.reply_text("📩 Texto recebido. A investigação começou.")
            resultado = await self.motor.analisar_texto(texto)
        except Exception as exc:
            logger.error("Erro ao analisar texto no privado (%s).", type(exc).__name__)
            await update.message.reply_text(
                "Ocorreu um erro durante a análise. Tente novamente mais tarde."
            )
            return

        relatorio = self._formatar_relatorio(resultado)
        await update.message.reply_text(relatorio, parse_mode="Markdown")

    async def _handle_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error(
            "Erro ao processar atualização do Telegram (%s).",
            type(context.error).__name__,
        )

    @staticmethod
    def _formatar_relatorio(resultado: ResultadoAnalise) -> str:
        partes = ["🔎 *Relatório de Investigação*\n"]
        if resultado.analise_linguistica:
            partes.append(f"*Análise da Linguagem:*\n{resultado.analise_linguistica}\n")
        if resultado.alegacoes:
            partes.append(
                "*Alegações identificadas:*\n"
                + "\n".join(f"• {alegacao}" for alegacao in resultado.alegacoes)
                + "\n"
            )
        if resultado.resultado_factual:
            partes.append(f"*Resultado Factual:*\n{resultado.resultado_factual}\n")
        if resultado.fontes:
            partes.append("*Fontes consultadas:*\n" + "\n".join(f"• {fonte}" for fonte in resultado.fontes))
        return "\n".join(partes)

    # ------------------------------------------------------------------ #
    #  Compat: API original                                               #
    # ------------------------------------------------------------------ #

    async def receber_texto(self, texto: str) -> str:
        """Encaminha uma mensagem textual ao motor central."""
        resultado = await self.motor.analisar_texto(texto)
        return (
            f"Análise da linguagem:\n{resultado.analise_linguistica}\n\n"
            f"Resultado factual:\n{resultado.resultado_factual}"
        )

    def run(self) -> None:
        """Inicia o polling da Telegram Bot API."""
        self.app.run_polling()



def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    BotTelegram().run()


if __name__ == "__main__":
    main()
