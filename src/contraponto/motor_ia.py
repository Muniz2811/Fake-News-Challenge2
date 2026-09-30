"""Estrutura do motor central do Divã Socrático."""

import logging
from dataclasses import dataclass, field

from claims import ExtratorAlegações, Alegacao
from search import (
    GoogleFactCheckAdapter,
    TavilySearchAdapter,
    WebSearchProvider,
    Evidencia,
)

logger = logging.getLogger(__name__)


@dataclass
class ResultadoAnalise:
    """Mantém análise linguística, alegações e evidências separadas."""

    analise_linguistica: str = ""
    alegacoes: list[Alegacao] = field(default_factory=list)
    evidencias: list[Evidencia] = field(default_factory=list)
    resultado_factual: str = ""
    requer_revisao: bool = False


class MotorIA:
    """Motor central; orquestra extração de alegações e busca de evidências."""

    def __init__(
        self,
        provedores_busca: list[WebSearchProvider] | None = None,
    ):
        self.extrator_claims = ExtratorAlegações()

        # Se nenhuma lista explícita for fornecida, usa os provedores padrão.
        if provedores_busca is not None:
            self.provedores_busca = provedores_busca
        else:
            self.provedores_busca: list[WebSearchProvider] = [
                GoogleFactCheckAdapter(),
                TavilySearchAdapter(),
            ]

    async def analisar_texto(self, texto: str) -> ResultadoAnalise:
        if not texto or not texto.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")

        # 1. Extração de alegações (MVP-06)
        resultado_claims = self.extrator_claims.extrair(texto)

        # 2. Busca de evidências por alegação individual (MVP-09)
        todas_evidencias: list[Evidencia] = []
        for alegacao in resultado_claims.alegacoes:
            for provedor in self.provedores_busca:
                try:
                    evs = provedor.buscar_evidencias_por_alegacao(alegacao.texto)
                    todas_evidencias.extend(evs)
                except Exception:
                    # Falha de um provedor nunca interrompe o pipeline
                    logger.exception(
                        "Provedor %s falhou para alegação: %.80s",
                        type(provedor).__name__,
                        alegacao.texto,
                    )

        return ResultadoAnalise(
            analise_linguistica="Análise inicial de linguagem concluída.",
            alegacoes=resultado_claims.alegacoes,
            evidencias=todas_evidencias,
            resultado_factual=resultado_claims.para_lista_numerada(),
            requer_revisao=False,
        )
