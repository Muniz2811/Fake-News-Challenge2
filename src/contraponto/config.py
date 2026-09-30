"""Módulo de configuração e validação de variáveis de ambiente."""

import os
from dotenv import load_dotenv

# Carrega as variáveis declaradas no arquivo .env
load_dotenv()

# Variáveis de ambiente para conexões externas
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Limite configurável de alegações por mensagem (Requisito MVP-06)
MAX_CLAIMS_PER_MESSAGE = int(os.getenv("MAX_CLAIMS_PER_MESSAGE", "5"))

# Opcional: lista de IDs de grupos autorizados (separados por vírgula).
# Se vazia ou não definida, o bot processa mensagens de qualquer grupo.
_authorized_groups_env = os.getenv("AUTHORIZED_GROUPS", "")
AUTHORIZED_GROUPS: list[int] = [
    int(g.strip()) for g in _authorized_groups_env.split(",") if g.strip()
]


def validate_env() -> None:
    """Garante que as variáveis de ambiente obrigatórias foram carregadas.

    Lança uma exceção explicativa caso falte alguma chave essencial.
    """
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise ValueError(
            "❌ [ERRO DE CONFIGURAÇÃO] Variáveis de ambiente obrigatórias ausentes:\n"
            f"   -> {', '.join(missing)}\n\n"
            "Por favor, crie um arquivo '.env' na raiz do projeto (baseado em '.env.example') "
            "e preencha com as suas chaves de API."
        )
