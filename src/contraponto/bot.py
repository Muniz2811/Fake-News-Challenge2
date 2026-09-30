"""Interface do Contraponto com o Telegram (python-telegram-bot ≥ 20)."""

import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Adiciona o diretório atual ao sys.path para permitir a importação direta dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TELEGRAM_BOT_TOKEN, AUTHORIZED_GROUPS, validate_env
from motor_ia import MotorIA, ResultadoAnalise

logger = logging.getLogger(__name__)

# Tempo de vida (em horas) das investigações armazenadas em memória.
_TTL_HORAS: int = 24


class BotTelegram:

    def __init__(
        self,
        motor: MotorIA,
        token: str,
        authorized_groups: list[int] | None = None,
    ) -> None:
        self.motor = motor
        self.authorized_groups: list[int] = authorized_groups or []

        # Armazena UUID → {resultado, expires}
        self._investigacoes: Dict[str, dict] = {}

        if not token:
            raise ValueError("Token do Telegram não fornecido ou inválido.")
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
    #  Armazenamento temporário com TTL                                   #
    # ------------------------------------------------------------------ #

    def _store(self, resultado: ResultadoAnalise) -> str:
        """Gera UUID, armazena resultado com TTL e faz limpeza oportuna."""
        self._purge_expired()
        analise_id = str(uuid.uuid4())
        self._investigacoes[analise_id] = {
            "resultado": resultado,
            "expires": datetime.now() + timedelta(hours=_TTL_HORAS),
        }
        return analise_id

    def _fetch(self, analise_id: str) -> Optional[ResultadoAnalise]:
        """Retorna o resultado se existir e não tiver expirado."""
        registro = self._investigacoes.get(analise_id)
        if not registro:
            return None
        if registro["expires"] < datetime.now():
            del self._investigacoes[analise_id]
            return None
        return registro["resultado"]

    def _purge_expired(self) -> None:
        agora = datetime.now()
        expirados = [k for k, v in self._investigacoes.items() if v["expires"] < agora]
        for k in expirados:
            del self._investigacoes[k]

    # ------------------------------------------------------------------ #
    #  Handlers                                                           #
    # ------------------------------------------------------------------ #

    async def _handle_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Analisa mensagens em grupos autorizados; alerta se requer revisão."""
        if not update.message or not update.message.text:
            return

        chat_id = update.message.chat_id

        # Se AUTHORIZED_GROUPS foi configurado, restringe.
        if self.authorized_groups and chat_id not in self.authorized_groups:
            return

        texto = update.message.text
        try:
            resultado = await self.motor.analisar_texto(texto)
        except NotImplementedError:
            logger.debug("Motor de análise ainda não implementado — ignorando.")
            return
        except Exception:
            logger.exception("Erro ao analisar texto no grupo %s", chat_id)
            return

        if not resultado.requer_revisao:
            return

        analise_id = self._store(resultado)

        bot_me = await context.bot.get_me()
        deep_link = f"https://t.me/{bot_me.username}?start={analise_id}"

        teclado = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔎 Ler relatório detalhado", url=deep_link)]]
        )

        mensagem_aviso = (
            "⚠️ Nossa IA identificou alegações nesta mensagem que podem "
            "requerer contexto adicional. Verifique os fatos antes de compartilhar."
        )

        await update.message.reply_text(mensagem_aviso, reply_markup=teclado)

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

        analise_id = context.args[0]
        resultado = self._fetch(analise_id)

        if resultado is None:
            await update.message.reply_text(
                "O link de investigação não é mais válido ou expirou.\n"
                "Se quiser uma nova análise, encaminhe o texto original "
                "diretamente para mim."
            )
            return

        relatorio = self._formatar_relatorio(resultado)
        await update.message.reply_text(relatorio, parse_mode="Markdown")

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
        except NotImplementedError:
            await update.message.reply_text(
                "O fluxo de análise ainda não está totalmente implementado."
            )
            return
        except Exception:
            logger.exception("Erro ao analisar texto no privado")
            await update.message.reply_text(
                "Ocorreu um erro durante a análise. Tente novamente mais tarde."
            )
            return

        relatorio = self._formatar_relatorio(resultado)
        await update.message.reply_text(relatorio, parse_mode="Markdown")

    async def _handle_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error("Erro ao processar atualização do Telegram: %s", context.error)

    # ------------------------------------------------------------------ #
    #  Formatação                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _formatar_relatorio(resultado: ResultadoAnalise) -> str:
        partes = ["🔎 *Relatório de Investigação*\n"]

        if resultado.analise_linguistica:
            partes.append(f"*Análise da Linguagem:*\n{resultado.analise_linguistica}\n")

        if resultado.alegacoes:
            lista = "\n".join(f"• {a}" for a in resultado.alegacoes)
            partes.append(f"*Alegações identificadas:*\n{lista}\n")

        if resultado.resultado_factual:
            partes.append(f"*Resultado Factual:*\n{resultado.resultado_factual}\n")

        if resultado.evidencias:
            lista = "\n".join(f"• {e.url}" for e in resultado.evidencias if e.url)
            partes.append(f"*Fontes consultadas:*\n{lista}")

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


def main() -> None:
    """Ponto de entrada: valida ambiente, instancia dependências e inicia polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("🔍 Validando variáveis de ambiente...")
    validate_env()
    print("✅ Configurações e variáveis de ambiente validadas com sucesso!")

    motor = MotorIA()
    bot = BotTelegram(
        motor=motor,
        token=TELEGRAM_BOT_TOKEN,
        authorized_groups=AUTHORIZED_GROUPS if AUTHORIZED_GROUPS else None,
    )

    print("🤖 Contraponto iniciado — aguardando mensagens...")
    bot.app.run_polling()


if __name__ == "__main__":
    main()
