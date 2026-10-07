"""Módulo de configuração e validação de variáveis de ambiente."""

import os
from dotenv import load_dotenv

# Carrega as variáveis declaradas no arquivo .env
load_dotenv()

# Variáveis de ambiente para conexões externas
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
# Opcional: sem chave, a extração usa regras e o verificador conserva a abstenção.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
GOOGLE_FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY", "").strip() or None
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip() or None
LANGUAGE_RISK_MODEL_PATH = os.getenv(
    "LANGUAGE_RISK_MODEL_PATH", "models/language_risk/baseline_tfidf_logreg.joblib"
)

# Limite configurável de alegações por mensagem (Requisito MVP-06)
MAX_CLAIMS_PER_MESSAGE = int(os.getenv("MAX_CLAIMS_PER_MESSAGE", "5"))

def parse_authorized_groups(value: str) -> list[int]:
    """Lê IDs explícitos; configuração vazia não autoriza nenhum grupo."""
    try:
        return list(dict.fromkeys(int(group.strip()) for group in value.split(",") if group.strip()))
    except ValueError:
        raise ValueError("AUTHORIZED_GROUPS deve conter IDs numéricos separados por vírgula.") from None


AUTHORIZED_GROUPS: list[int] = parse_authorized_groups(os.getenv("AUTHORIZED_GROUPS", ""))


def validate_env() -> None:
    """Garante que as variáveis de ambiente obrigatórias foram carregadas.

    Lança uma exceção explicativa caso falte alguma chave essencial.
    """
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if missing:
        raise ValueError(
            "❌ [ERRO DE CONFIGURAÇÃO] Variáveis de ambiente obrigatórias ausentes:\n"
            f"   -> {', '.join(missing)}\n\n"
            "Por favor, crie um arquivo '.env' na raiz do projeto (baseado em '.env.example') "
            "e preencha com as suas chaves de API."
        )
