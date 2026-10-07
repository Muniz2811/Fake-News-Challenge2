#!/usr/bin/env python3
"""Avaliação local pequena e reproduzível do checkpoint configurado no Ollama."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "contraponto"))

from claims import Alegacao, ExtratorAlegações  # noqa: E402
from config import LLM_BASE_URL, LLM_MODEL  # noqa: E402
from llm import criar_provedor_llm  # noqa: E402
from search import Evidencia  # noqa: E402
from socratic import GerenciadorSocratico  # noqa: E402
from verifier import VerificadorAlegacoes  # noqa: E402


class ProvedorObservavel:
    """Registra as saídas estruturadas para tornar a avaliação auditável."""

    def __init__(self, provedor):
        self.provedor = provedor
        self.saidas = []

    def gerar_json(self, sistema, entrada, schema, nome):
        print(f"Avaliando {nome}...", file=sys.stderr, flush=True)
        try:
            saida = self.provedor.gerar_json(sistema, entrada, schema, nome)
            self.saidas.append({"tarefa": nome, "saida": saida})
            return saida
        except Exception as erro:
            self.saidas.append({"tarefa": nome, "erro": type(erro).__name__})
            raise


def main() -> int:
    llm = criar_provedor_llm()
    if llm is None:
        print("Configure LLM_BASE_URL, por exemplo http://localhost:11434/v1.")
        return 2

    llm = ProvedorObservavel(llm)
    inicio = time.monotonic()
    texto = (
        "A prefeitura informou que 20 escolas serão inauguradas em dezembro. "
        "Na minha opinião, isso é maravilhoso. Você pode compartilhar?"
    )
    extracao = ExtratorAlegações(limit=3, llm=llm).extrair(texto)
    extracao_ok = (
        len(extracao.alegacoes) == 1
        and "20" in extracao.alegacoes[0].texto
        and "opinião" not in extracao.alegacoes[0].trecho_origem.casefold()
    )

    alegacao = Alegacao(1, "A prefeitura inaugurará 20 escolas em dezembro.")
    apoio = Evidencia(
        title="Plano oficial", url="https://example.org/plano", publisher="Prefeitura",
        snippet="O plano oficial prevê a inauguração de 20 escolas em dezembro.",
        source_type="official",
    )
    refutacao = Evidencia(
        title="Revisão do cronograma", url="https://example.org/revisao", publisher="Secretaria",
        snippet="A prefeitura não inaugurará 20 escolas em dezembro; o cronograma foi adiado.",
        source_type="official",
    )
    irrelevante = Evidencia(
        title="Previsão do tempo", url="https://example.org/tempo", publisher="Instituto",
        snippet="Dezembro terá chuvas acima da média no município.", source_type="official",
    )
    verificador = VerificadorAlegacoes(llm=llm)
    categorias = {
        "apoio": verificador.verificar_alegacao(alegacao, [apoio]).categoria,
        "refutacao": verificador.verificar_alegacao(alegacao, [refutacao]).categoria,
        "irrelevante": verificador.verificar_alegacao(alegacao, [irrelevante]).categoria,
        "conflito": verificador.verificar_alegacao(alegacao, [apoio, refutacao]).categoria,
    }
    nli_ok = categorias == {
        "apoio": "apoiada",
        "refutacao": "refutada",
        "irrelevante": "evidencia_insuficiente",
        "conflito": "conflitante",
    }

    veredito = verificador.verificar_alegacao(alegacao, [apoio, refutacao])
    dialogo = GerenciadorSocratico(llm=llm)
    mensagens = [dialogo.iniciar_sessao(1, texto, [veredito], chat_id=1)]
    mensagens.append(dialogo.processar_resposta(1, "Eu conferiria a fonte oficial.", 1))
    mensagens.append(dialogo.processar_resposta(1, "Os cronogramas são diferentes.", 1))
    mensagens.append(dialogo.processar_resposta(1, "Ainda há incerteza.", 1))
    dialogo_ok = (
        "Pergunta 1/3" in mensagens[0]
        and "Você mencionou" in mensagens[1]
        and "Pergunta 3/3" in mensagens[2]
        and "Síntese" in mensagens[3]
        and not dialogo.tem_sessao_ativa(1, 1)
    )
    llm_sem_fallback = len(llm.saidas) == 8 and all(
        "saida" in registro for registro in llm.saidas
    )

    resultado = {
        "endpoint": LLM_BASE_URL,
        "checkpoint": LLM_MODEL,
        "duracao_segundos": round(time.monotonic() - inicio, 2),
        "extracao": {
            "ok": extracao_ok,
            "alegacoes": [a.texto for a in extracao.alegacoes],
        },
        "nli": {"ok": nli_ok, "categorias": categorias},
        "dialogo": {"ok": dialogo_ok, "mensagens_geradas": len(mensagens)},
        "llm_sem_fallback": llm_sem_fallback,
        "saidas_estruturadas": llm.saidas,
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0 if extracao_ok and nli_ok and dialogo_ok and llm_sem_fallback else 1


if __name__ == "__main__":
    raise SystemExit(main())
