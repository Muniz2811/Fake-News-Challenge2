"""Interface do Contraponto com o Telegram (python-telegram-bot ≥ 20)."""

import asyncio
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Adiciona o diretório atual ao sys.path para permitir a importação direta dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    TELEGRAM_BOT_TOKEN, AUTHORIZED_GROUPS, LLM_API_KEY, OPENAI_API_KEY,
    GOOGLE_FACT_CHECK_KEY, TAVILY_API_KEY, validate_env,
)
from motor_ia import MotorIA, ResultadoAnalise
from verifier import ROTULOS_CATEGORIAS, VereditoAlegacao
from socratic import GerenciadorSocratico

logger = logging.getLogger(__name__)

# Tempo de vida (em horas) das investigações armazenadas em memória.
_TTL_HORAS: int = 24
_MAX_INVESTIGACOES: int = 1000

# Limite do Telegram por mensagem. Contado em unidades UTF-16 (emojis fora do
# BMP valem 2), o que nunca subestima o tamanho real.
_LIMITE_MENSAGEM_TELEGRAM: int = 4096

_SEPARADOR_SECAO = "━━━━━━━━━━━━━━━━"
_AVISO_LINGUAGEM = (
    "ℹ️ Esta seção mede apenas sinais do texto (estilo e padrões de escrita). "
    "Ela não avalia se as informações são verdadeiras ou falsas."
)
_AVISO_FACTUAL = (
    "ℹ️ Cada alegação foi comparada separadamente com as fontes encontradas. "
    "O resultado factual não leva em conta a análise da linguagem."
)
_LIMITACOES_POR_CATEGORIA = {
    "nao_verificada": (
        "não foi possível concluir a verificação com segurança. Recomendamos revisão manual."
    ),
    "evidencia_insuficiente": (
        "as fontes encontradas não permitem uma conclusão segura. Recomendamos revisão manual."
    ),
    "conflitante": "as fontes consultadas divergem entre si. Recomendamos revisão manual.",
}
# Identificadores internos de evidência (sha256) não ajudam o leitor no celular.
_ID_EVIDENCIA = re.compile(r"^\[ev_[0-9a-f]{64}\] ", re.MULTILINE)


def _tamanho_telegram(texto: str) -> int:
    return len(texto.encode("utf-16-le")) // 2


def _cortar_por_tamanho(texto: str, limite: int) -> list[str]:
    """Último recurso: corta uma linha maior que o limite sem quebrar caracteres."""
    pedacos: list[str] = []
    atual: list[str] = []
    tamanho = 0
    for caractere in texto:
        tamanho_caractere = _tamanho_telegram(caractere)
        if tamanho + tamanho_caractere > limite:
            pedacos.append("".join(atual))
            atual, tamanho = [], 0
        atual.append(caractere)
        tamanho += tamanho_caractere
    if atual:
        pedacos.append("".join(atual))
    return pedacos


class BotTelegram:

    def __init__(
        self,
        motor: MotorIA | None = None,
        token: str | None = None,
        authorized_groups: list[int] | None = None,
    ) -> None:
        self.motor = motor if motor is not None else MotorIA()
        gerenciador = getattr(self.motor, "socratico", None)
        self.socratico = gerenciador if gerenciador is not None else GerenciadorSocratico()
        self.authorized_groups = list(AUTHORIZED_GROUPS if authorized_groups is None else authorized_groups)
        token = TELEGRAM_BOT_TOKEN if token is None else token

        # Armazena UUID → {resultado, expires}
        self._investigacoes: Dict[str, dict] = {}
        self._dialogos_iniciando: set[tuple[int, int, str]] = set()

        if not token:
            raise ValueError("Token do Telegram não fornecido ou inválido.")
        self.app = ApplicationBuilder().token(token).build()

        # /start com deep-link (chat privado)
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("ajuda", self._cmd_ajuda))
        self.app.add_handler(CommandHandler("chatid", self._cmd_chatid))
        self.app.add_handler(CommandHandler("cancelar", self._cmd_cancelar))
        self.app.add_handler(CallbackQueryHandler(
            self._iniciar_dialogo_socratico, pattern=r"^socratico:", block=False,
        ))

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
        if not hasattr(self, "_investigacoes"):
            self._investigacoes = {}
        self._purge_expired()
        analise_id = str(uuid.uuid4())
        self._investigacoes[analise_id] = {
            "resultado": resultado,
            "expires": datetime.now() + timedelta(hours=_TTL_HORAS),
        }
        while len(self._investigacoes) > _MAX_INVESTIGACOES:
            self._investigacoes.pop(next(iter(self._investigacoes)))
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
        if (
            not update.message or not update.message.text or not update.message.text.strip()
            or update.message.chat.type not in {"group", "supergroup"}
        ):
            return

        chat_id = update.message.chat_id

        # Falha fechada: lista vazia não autoriza qualquer grupo.
        if chat_id not in self.authorized_groups:
            return

        texto = update.message.text
        try:
            resultado = await self.motor.analisar_texto(texto)
        except NotImplementedError:
            logger.debug("Motor de análise ainda não implementado — ignorando.")
            return
        except Exception as exc:
            logger.error("Erro ao analisar texto no grupo (%s).", type(exc).__name__)
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

    async def _cmd_chatid(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mostra apenas o ID do grupo atual, sem analisar ou registrar mensagens."""
        if not update.message:
            return
        if update.message.chat.type in {"group", "supergroup"}:
            await update.message.reply_text(f"ID deste grupo: {update.message.chat_id}")
        else:
            await update.message.reply_text("Use /chatid dentro do grupo que deseja autorizar.")

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

        await self._enviar_relatorio(update.message, resultado, analise_id)

    async def _cmd_ajuda(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message or update.message.chat.type != "private":
            return
        await update.message.reply_text(
            "Envie uma notícia ou alegação como texto para iniciar uma investigação. "
            "Áudios, figurinhas e arquivos ainda não são aceitos. "
            "No grupo, use /chatid para consultar o ID e cadastre-o em AUTHORIZED_GROUPS."
        )

    @staticmethod
    def _ids_privados(update: Update) -> tuple[int | None, int | None]:
        usuario = getattr(update, "effective_user", None)
        chat = getattr(update, "effective_chat", None)
        message = getattr(update, "message", None)
        if usuario is None and message is not None:
            usuario = getattr(message, "from_user", None)
        if chat is None and message is not None:
            chat = getattr(message, "chat", None)
        return getattr(usuario, "id", None), getattr(chat, "id", None)

    async def _cmd_cancelar(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        del context
        if not update.message or update.message.chat.type != "private":
            return
        user_id, chat_id = self._ids_privados(update)
        if user_id is None or chat_id is None:
            return
        await update.message.reply_text(self.socratico.cancelar_sessao(user_id, chat_id))

    async def _iniciar_dialogo_socratico(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        del context
        query = getattr(update, "callback_query", None)
        if query is None:
            return
        try:
            await query.answer("Preparando o diálogo…")
        except BadRequest as exc:
            # Um callback expira rapidamente no Telegram. Isso pode ocorrer quando
            # vários cliques antigos chegam depois de uma tarefa demorada.
            logger.warning("Callback do diálogo recusado pelo Telegram: %s", str(exc)[:160])
            return
        data = getattr(query, "data", "") or ""
        analise_id = data.removeprefix("socratico:")
        resultado = self._fetch(analise_id)
        if resultado is None:
            await query.message.reply_text(
                "Esta investigação expirou. Envie o texto novamente para gerar outro relatório."
            )
            return
        usuario = getattr(query, "from_user", None)
        chat = getattr(query.message, "chat", None)
        user_id, chat_id = getattr(usuario, "id", None), getattr(chat, "id", None)
        if user_id is None or chat_id is None or getattr(chat, "type", None) != "private":
            await query.message.reply_text("O diálogo socrático está disponível no chat privado.")
            return
        if not hasattr(self, "_dialogos_iniciando"):
            self._dialogos_iniciando = set()
        chave = (chat_id, user_id, analise_id)
        if chave in self._dialogos_iniciando:
            await query.message.reply_text(
                "Já estou preparando a primeira pergunta deste diálogo."
            )
            return
        if self.socratico.tem_sessao_ativa(user_id, chat_id):
            await query.message.reply_text(
                "Este diálogo já está em andamento. Responda à pergunta anterior "
                "ou use /cancelar para encerrá-lo."
            )
            return

        self._dialogos_iniciando.add(chave)
        try:
            # O Ollama é síncrono. Executá-lo em uma thread evita congelar o loop
            # do Telegram enquanto a pergunta é preparada.
            abertura = await asyncio.to_thread(
                self.socratico.iniciar_sessao,
                user_id,
                resultado.texto_original,
                resultado.vereditos,
                chat_id=chat_id,
            )
            await query.message.reply_text(abertura)
        finally:
            self._dialogos_iniciando.discard(chave)

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

        user_id, chat_id = self._ids_privados(update)
        if (
            user_id is not None and chat_id is not None
            and self.socratico.tem_sessao_ativa(user_id, chat_id)
        ):
            await update.message.reply_text(
                "Estou relacionando sua resposta com o que foi encontrado…"
            )
            resposta = await asyncio.to_thread(
                self.socratico.processar_resposta, user_id, texto, chat_id,
            )
            await update.message.reply_text(resposta)
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
        except Exception as exc:
            logger.error("Erro ao analisar texto no privado (%s).", type(exc).__name__)
            await update.message.reply_text(
                "Ocorreu um erro durante a análise. Tente novamente mais tarde."
            )
            return

        analise_id = self._store(resultado)
        await self._enviar_relatorio(update.message, resultado, analise_id)

    async def _handle_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        erro = context.error
        if isinstance(erro, BadRequest):
            logger.error(
                "Erro ao processar atualização do Telegram (BadRequest): %s",
                str(erro)[:200],
            )
            return
        logger.error("Erro ao processar atualização do Telegram (%s).", type(erro).__name__)

    # ------------------------------------------------------------------ #
    #  Formatação                                                         #
    # ------------------------------------------------------------------ #

    async def _enviar_relatorio(
        self, message, resultado: ResultadoAnalise, analise_id: str | None = None,
    ) -> None:
        """Envia o relatório em uma ou mais mensagens, na ordem das seções."""
        partes = self._dividir_mensagem(self._formatar_relatorio(resultado))
        for indice, parte in enumerate(partes):
            kwargs = {}
            if analise_id and indice == len(partes) - 1:
                kwargs["reply_markup"] = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        "💬 Iniciar diálogo socrático",
                        callback_data=f"socratico:{analise_id}",
                    )
                ]])
            await message.reply_text(parte, **kwargs)

    @staticmethod
    def _formatar_relatorio(resultado: ResultadoAnalise) -> str:
        # Texto simples: alegações, justificativas e URLs podem conter caracteres
        # de Markdown. Não submeter conteúdo dinâmico ao parser do Telegram.
        # URLs em texto simples continuam clicáveis no Telegram.
        secoes = [
            "🔎 Relatório de Investigação",
            BotTelegram._secao_linguagem(resultado),
            BotTelegram._secao_factual(resultado),
        ]
        if resultado.etapas_indisponiveis:
            avisos = "\n".join(f"• {etapa}" for etapa in resultado.etapas_indisponiveis)
            secoes.append(f"{_SEPARADOR_SECAO}\n⚠️ Etapas indisponíveis\n{avisos}")
        return "\n\n".join(secoes)

    @staticmethod
    def _secao_linguagem(resultado: ResultadoAnalise) -> str:
        analise = resultado.analise_linguistica or "Análise linguística indisponível."
        return f"{_SEPARADOR_SECAO}\n1️⃣ Análise da linguagem\n{_AVISO_LINGUAGEM}\n\n{analise}"

    @staticmethod
    def _secao_factual(resultado: ResultadoAnalise) -> str:
        partes = [f"{_SEPARADOR_SECAO}\n2️⃣ Verificação factual por alegação\n{_AVISO_FACTUAL}"]
        if resultado.vereditos:
            partes.extend(BotTelegram._bloco_alegacao(v) for v in resultado.vereditos)
            return "\n\n".join(partes)

        # Sem vereditos individuais: preserva o conteúdo agregado disponível.
        if resultado.alegacoes:
            lista = "\n".join(f"{a.id}. {a.texto}" for a in resultado.alegacoes)
            partes.append(f"Alegações identificadas:\n{lista}")
            partes.append(resultado.resultado_factual or "Resultado factual indisponível.")
        else:
            partes.append(
                resultado.resultado_factual
                or "Nenhuma alegação verificável foi identificada no texto."
            )
        urls = [e.url for e in resultado.evidencias if e.url]
        if urls:
            partes.append("Fontes consultadas:\n" + "\n".join(urls))
        return "\n\n".join(partes)

    @staticmethod
    def _bloco_alegacao(veredito: VereditoAlegacao) -> str:
        rotulo = ROTULOS_CATEGORIAS.get(veredito.categoria, veredito.categoria)
        justificativa = (
            _ID_EVIDENCIA.sub("", veredito.justificativa).strip()
            or "Justificativa indisponível."
        )
        urls = [url for url in veredito.fontes_citadas if url]
        fontes = "\n".join(urls) if urls else "Nenhuma fonte disponível para esta alegação."
        linhas = [
            f"📌 Alegação {veredito.alegacao_id}",
            f'"{veredito.texto_alegacao}"',
            f"Resultado factual: {rotulo}",
            f"Justificativa:\n{justificativa}",
            f"Fontes:\n{fontes}",
        ]
        limitacao = _LIMITACOES_POR_CATEGORIA.get(veredito.categoria)
        if limitacao:
            linhas.append(f"Limitações: {limitacao}")
        return "\n".join(linhas)

    @staticmethod
    def _dividir_mensagem(
        texto: str, limite: int = _LIMITE_MENSAGEM_TELEGRAM
    ) -> list[str]:
        """Divide o texto em partes ≤ limite, preferindo quebrar entre parágrafos,
        depois entre linhas e, só em último caso, dentro de uma linha."""

        def dividir(bloco: str, separadores: list[str]) -> list[str]:
            if _tamanho_telegram(bloco) <= limite:
                return [bloco]
            if not separadores:
                return _cortar_por_tamanho(bloco, limite)
            separador, *restantes = separadores
            pedacos: list[str] = []
            atual = ""
            for parte in bloco.split(separador):
                for sub in dividir(parte, restantes):
                    candidato = f"{atual}{separador}{sub}" if atual else sub
                    if _tamanho_telegram(candidato) <= limite:
                        atual = candidato
                    else:
                        if atual:
                            pedacos.append(atual)
                        atual = sub
            if atual:
                pedacos.append(atual)
            return pedacos

        return [p.strip() for p in dividir(texto, ["\n\n", "\n"]) if p.strip()]

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


class _RedactingFormatter(logging.Formatter):
    """Remove credenciais inclusive de mensagens de erro e tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        texto = super().format(record)
        texto = re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot[CREDENCIAL OCULTA]", texto)
        for chave in (
            TELEGRAM_BOT_TOKEN, LLM_API_KEY, OPENAI_API_KEY,
            GOOGLE_FACT_CHECK_KEY, TAVILY_API_KEY,
        ):
            if chave:
                texto = texto.replace(chave, "[CREDENCIAL OCULTA]")
        return texto


def _configurar_logging() -> None:
    """Mantém logs da aplicação sem os URLs autenticados das requisições HTTP."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    formatter = _RedactingFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    """Ponto de entrada: valida ambiente, instancia dependências e inicia polling."""
    _configurar_logging()

    print("🔍 Validando variáveis de ambiente...")
    validate_env()
    print("✅ Configurações e variáveis de ambiente validadas com sucesso!")

    motor = MotorIA()
    bot = BotTelegram(
        motor=motor,
        token=TELEGRAM_BOT_TOKEN,
        authorized_groups=AUTHORIZED_GROUPS,
    )

    print("🤖 Contraponto iniciado — aguardando mensagens...")
    bot.app.run_polling()


if __name__ == "__main__":
    main()
