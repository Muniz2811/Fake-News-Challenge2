"""Estrutura do motor central do Divã Socrático."""

from dataclasses import dataclass, field
import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claims import ExtratorAlegações, Alegacao
from search import ProvedorBuscaWeb, Evidencia
from verifier import VerificadorAlegacoes, VereditoAlegacao
from socratic import GerenciadorSocratico

logger = logging.getLogger(__name__)

_NOMES_PROVEDORES = {
    "GoogleFactCheckAdapter": "Google Fact Check",
    "TavilySearchAdapter": "Tavily",
}

ROTULOS_CATEGORIAS = {
    "apoiada": "Apoiada por evidências",
    "refutada": "Refutada por evidências",
    "evidencia_insuficiente": "Evidência insuficiente",
    "conflitante": "Informação conflitante",
    "nao_verificada": "Não verificada",
}


@dataclass
class ResultadoAnalise:
    """Mantém análise linguística, alegações, evidências e vereditos."""

    analise_linguistica: str = ""
    alegacoes: list[Alegacao] = field(default_factory=list)
    evidencias: list[Evidencia] = field(default_factory=list)
    vereditos: list[VereditoAlegacao] = field(default_factory=list)
    resultado_factual: str = ""
    requer_revisao: bool = False
    score_linguistico: float | None = None
    etapas_indisponiveis: list[str] = field(default_factory=list)


class MotorIA:

    def __init__(self):
        self.extrator_claims = ExtratorAlegações()
        self.provedor_busca = ProvedorBuscaWeb()
        self.provedores_busca = [self.provedor_busca]
        self.verificador = VerificadorAlegacoes()
        self.socratico = GerenciadorSocratico()

    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")

        # 1. Extração de alegações (MVP-06)
        resultado_claims = self.extrator_claims.extrair(texto)

        todas_evidencias: list[Evidencia] = []
        lista_vereditos: list[VereditoAlegacao] = []
        indisponiveis: list[str] = []

        # 2. Busca de evidências (MVP-09) e Verificação (MVP-10)
        for alegacao in resultado_claims.alegacoes:
            evs_alegacao: list[Evidencia] = []
            for provedor in self.provedores_busca:
                try:
                    evs_alegacao.extend(
                        provedor.buscar_evidencias_por_alegacao(alegacao.texto)
                    )
                except Exception:
                    logger.warning("Provedor %s indisponível.", type(provedor).__name__)
                    nome = _NOMES_PROVEDORES.get(
                        type(provedor).__name__, type(provedor).__name__
                    )
                    aviso = f"Fonte indisponível: a busca no {nome} não respondeu."
                    if aviso not in indisponiveis:
                        indisponiveis.append(aviso)

            todas_evidencias.extend(evs_alegacao)

            veredito = self.verificador.verificar_alegacao(alegacao, evs_alegacao)
            lista_vereditos.append(veredito)

        factual = "\n\n".join(
            f"{v.alegacao_id}. {v.texto_alegacao}\n"
            f"Conclusão: {ROTULOS_CATEGORIAS.get(v.categoria, v.categoria)}\n{v.justificativa}"
            for v in lista_vereditos
        ) or "Nenhuma alegação verificável foi identificada no texto."

        return ResultadoAnalise(
            analise_linguistica="Análise de linguagem concluída.",
            alegacoes=resultado_claims.alegacoes,
            evidencias=todas_evidencias,
            vereditos=lista_vereditos,
            resultado_factual=factual,
            requer_revisao=False,
            score_linguistico=None,
            etapas_indisponiveis=indisponiveis,
        )