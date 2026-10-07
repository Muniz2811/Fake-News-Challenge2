"""Adaptador pequeno para LLMs compatíveis com a API da OpenAI, como Ollama."""

import json
from typing import Any, Protocol

from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
)


class ProvedorLLM(Protocol):
    """Contrato independente do modelo usado pelo núcleo da aplicação."""

    def gerar_json(
        self, sistema: str, entrada: str, schema: dict[str, Any], nome: str
    ) -> dict[str, Any]: ...


class LLMCompativelOpenAI:
    """Solicita JSON estruturado a qualquer endpoint compatível com Chat Completions."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def gerar_json(
        self, sistema: str, entrada: str, schema: dict[str, Any], nome: str
    ) -> dict[str, Any]:
        resposta = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": entrada},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": nome, "strict": True, "schema": schema},
            },
            # O Ollama aceita este campo e evita o raciocínio longo do Qwen 3.
            extra_body={"think": False},
        )
        escolha = resposta.choices[0]
        if getattr(escolha, "finish_reason", None) != "stop":
            raise ValueError("Resposta incompleta da LLM")
        conteudo = escolha.message.content
        if not isinstance(conteudo, str):
            raise ValueError("Resposta vazia da LLM")
        dados = json.loads(conteudo)
        if not isinstance(dados, dict):
            raise ValueError("A resposta estruturada deve ser um objeto")
        return dados


def criar_provedor_llm() -> ProvedorLLM | None:
    """Cria o provedor configurado; sem configuração, preserva os fallbacks locais."""
    if LLM_BASE_URL:
        cliente = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY or "ollama",
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        return LLMCompativelOpenAI(cliente, LLM_MODEL or "qwen3:8b")
    if OPENAI_API_KEY:
        cliente = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        return LLMCompativelOpenAI(cliente, LLM_MODEL or "gpt-4o-mini")
    return None
