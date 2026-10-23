"""Módulo de configuração e validação de variáveis de ambiente."""

import os
from dotenv import load_dotenv

# Carrega as variáveis declaradas no arquivo .env
load_dotenv()

# Variáveis de ambiente para conexões externas
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
# Opcional: sem chave, a extração usa regras e o verificador conserva a abstenção.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip() or None
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or None
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
# Qwen 3 pode gastar parte do orçamento interno antes de produzir o JSON.
# 4.096 preserva espaço para uma resposta estruturada completa com até quatro
# evidências; o limite continua configurável por ambiente.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
# Quatro fontes preservam diversidade suficiente para a comparação sem inflar
# demais o contexto da LLM. O valor continua configurável no ambiente.
MAX_EVIDENCES_PER_CLAIM = int(os.getenv("MAX_EVIDENCES_PER_CLAIM", "4"))
# A Tavily costuma devolver trechos de cerca de mil caracteres. Preservamos esse
# contexto para a comparação, sem obrigar o usuário a configurar a variável.
MAX_EVIDENCE_SNIPPET_CHARS = int(os.getenv("MAX_EVIDENCE_SNIPPET_CHARS", "1200"))
# O relatório mostra uma frase relevante e pode exibir mais contexto sob demanda.
MAX_EVIDENCE_REPORT_SNIPPET_CHARS = int(
    os.getenv("MAX_EVIDENCE_REPORT_SNIPPET_CHARS", "500")
)
GOOGLE_FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY", "").strip() or None
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip() or None
LANGUAGE_RISK_MODEL_PATH = os.getenv(
    "LANGUAGE_RISK_MODEL_PATH", "models/language_risk/baseline_tfidf_logreg.joblib"
)
LANGUAGE_RISK_BACKEND = os.getenv("LANGUAGE_RISK_BACKEND", "tfidf").strip().lower()
BERTIMBAU_MLP_METRICS_PATH = os.getenv(
    "BERTIMBAU_MLP_METRICS_PATH",
    "data/processed/language_risk/bertimbau/metrics_mlp.json",
)
BERTIMBAU_LOGREG_METRICS_PATH = os.getenv(
    "BERTIMBAU_LOGREG_METRICS_PATH",
    "data/processed/language_risk/bertimbau/metrics.json",
)
BERTIMBAU_MODEL_DIR = os.getenv(
    "BERTIMBAU_MODEL_DIR", "models/language_risk/bertimbau_base"
)
BERTIMBAU_DEVICE = os.getenv("BERTIMBAU_DEVICE", "auto").strip().lower()

# Limite configurável de alegações por mensagem (Requisito MVP-06)
MAX_CLAIMS_PER_MESSAGE = int(os.getenv("MAX_CLAIMS_PER_MESSAGE", "5"))

def parse_authorized_groups(value: str) -> list[int]:
    """Lê IDs explícitos; configuração vazia não autoriza nenhum grupo."""
    try:
        return list(dict.fromkeys(int(group.strip()) for group in value.split(",") if group.strip()))
    except ValueError:
        raise ValueError("AUTHORIZED_GROUPS deve conter IDs numéricos separados por vírgula.") from None


AUTHORIZED_GROUPS: list[int] = parse_authorized_groups(os.getenv("AUTHORIZED_GROUPS", ""))
# Comando explícito por padrão; o modo automático preserva alerta + deep link.
GROUP_ANALYSIS_MODE = os.getenv("GROUP_ANALYSIS_MODE", "command").strip().lower()


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
    if LANGUAGE_RISK_BACKEND not in {"tfidf", "bertimbau_logreg", "bertimbau_mlp"}:
        raise ValueError(
            "LANGUAGE_RISK_BACKEND deve ser tfidf, bertimbau_logreg ou bertimbau_mlp"
        )
    if BERTIMBAU_DEVICE not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("BERTIMBAU_DEVICE deve ser auto, cpu, mps ou cuda")
    if GROUP_ANALYSIS_MODE not in {"command", "automatic"}:
        raise ValueError("GROUP_ANALYSIS_MODE deve ser command ou automatic")
